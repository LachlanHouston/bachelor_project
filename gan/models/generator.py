from gan.models.DPRNN import DPRNN
import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as L
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from gan.utils.utils import stft_to_waveform, waveform_to_stft, perfect_shuffle
import wandb
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE

def return_waveform(x):
    x_real = x[:, 0, :, :]
    x_img  = x[:, 1, :, :]
    x_complex = torch.complex(x_real, x_img)
    x_waveform = torch.istft(x_complex, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))
    return x_waveform

def _padded_cat(x, y, dim=1):
    # Pad x to have same size with y, and cat them
    # x dim: N, C, T, F
    x_pad = F.pad(x, (0, y.shape[3] - x.shape[3], 
                      0, y.shape[2] - x.shape[2])) # pad T, F
    z = torch.cat((x_pad, y), dim=dim) # cat on C
    return z

class ConvBlock(nn.Module):
    "norm: weight, batch, layer, instance"
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), 
                 padding=(0, 1), causal=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.causal = causal        

    def forward(self, x):
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1),
                 padding=(0, 0), output_padding=(0, 0), is_last=False, causal=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                       padding, output_padding)        
        self.norm = nn.BatchNorm2d(out_channels)
        self.is_last = is_last
        self.causal = causal
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, param=None, in_channels=2, out_channels=2, visualize=False, return_waveform=True):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.rnn_block = DPRNN(128, rnn_type='LSTM', hidden_size=128, output_size=128, num_layers=2, bidirectional=True)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.visualize = visualize
        self.return_waveform = return_waveform

        # Encoder
        self.encoder.append(ConvBlock(self.in_channels, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))
        self.encoder.append(ConvBlock(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))
        self.encoder.append(ConvBlock(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))
        # self.encoder.append(ConvBlock(128, 256, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))

        # Decoder
        # self.decoder.append(TransConvBlock(512, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)))
        self.decoder.append(TransConvBlock(256, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)))
        self.decoder.append(TransConvBlock(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)))
        self.decoder.append(TransConvBlock(64, self.out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(0, 0), is_last=True))

        self.activation = nn.Tanh()

    def forward(self, x):
        if (isinstance(x, tuple) or isinstance(x, list)) and len(x[0].shape) == 4:
            x = torch.stack(x, dim=0).squeeze()
            print("True")
        e = x
        e_list = []
        """Encoder"""
        for i, layer in enumerate(self.encoder):
            # apply convolutional layer
            e = layer(e)
            # store the output for skip connection
            e_list.append(e)
            # store the feature maps for visualization
        
        """Dual-Path RNN"""
        rnn_out = self.rnn_block(e) # [32, 128, 32, 321]
        # store length to go through the list backwards
        idx = len(e_list)
        d = rnn_out

        """Decoder"""
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # concatenate d with the skip connection and put though layer
            d = layer(_padded_cat(d, e_list[idx]))
            # store the feature maps for visualization

        d = self.activation(d)
        
        # Perform hadamard product
        mask = d
        output = torch.mul(x, mask)

        if self.return_waveform:
            output = return_waveform(output).detach().cpu().numpy()
            mask = return_waveform(mask).detach().cpu().numpy()

        return output, mask
    
# Pytorch Lightning Module
class pl_Generator(L.LightningModule):
    def __init__(self, param=None, in_channels=2, out_channels=2, visualize=False, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.model = Generator(param, in_channels, out_channels, visualize)
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        real_clean = batch[0]
        real_noisy = batch[1]

        # Forward pass
        fake_clean, mask = self.model(real_noisy)

        fidelity = torch.norm(real_clean - fake_clean, p=1)

        # Loss
        loss = self.alpha_fidelity * fidelity

        self.log('Fidelity Loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.g_learning_rate)
        return optimizer

    def validation_step(self, batch, batch_idx):
        real_clean = batch[0]
        real_noisy = batch[1]

        # Forward pass
        fake_clean, mask = self.model(real_noisy)

        fidelity = torch.norm(real_clean - fake_clean, p=1)

        # Loss
        loss = self.alpha_fidelity * fidelity

        self.log('Validation Fidelity Loss', loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device)
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device)
        real_noisy_waveforms = stft_to_waveform(real_noisy, device=self.device)

        # Compute SI-SNR
        sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
        sisnr_score = sisnr(fake_clean_waveforms, real_clean_waveforms)
        self.log('Validation SI-SNR score', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Mean Opinion Score (SQUIM)
        if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
            reference_waveforms = perfect_shuffle(real_clean_waveforms)
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
            self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if batch_idx == 0:
            vis_idx = torch.randint(0, self.batch_size, (1,)).item()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(fake_clean_waveforms[vis_idx], sample_rate=16000, caption="Generated Clean Audio")]})
            self.logger.experiment.log({"real_clean_waveform": [wandb.Audio(real_clean_waveforms[vis_idx], sample_rate=16000, caption="Real Clean Audio")]})
            self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(real_noisy_waveforms[vis_idx], sample_rate=16000, caption="Real Noisy Audio")]})

        return loss

    
def visualize_feature_maps(model, input):
    # Visualize the feature maps of the model
    # Put input through the model and save every layer output
    feature_maps = []
    with torch.no_grad():
        _, _, maps = model(input)
        
        # Take the mean of the channel dimension
        for layer in maps:
            layer = layer.squeeze(0)
            feature_maps.append(layer.mean(dim=0))

    return feature_maps

if __name__ == '__main__':
    # print(torch.cuda.is_available())

    # Dummy train_loader
    train_loader = torch.utils.data.DataLoader(
        [torch.randn(4, 2, 257, 321), torch.randn(4, 2, 257, 321)],
        batch_size=4,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        [torch.randn(4, 2, 257, 321), torch.randn(4, 2, 257, 321)],
        batch_size=4,
        shuffle=False
    )

    # model = pl_Generator(batch_size=4, g_learning_rate=1e-4, 
    #                          alpha_fidelity=10)
    
    # trainer = L.Trainer(max_epochs=5, accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0,
    #                     log_every_n_steps=1, limit_train_batches=20, limit_val_batches=0,
    #                     logger=False,
    #                     fast_dev_run=False)
    # trainer.fit(model, train_loader, val_loader)

    model = Generator()

    # Get one batch and put through the model
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        real_clean = batch[0]
        real_noisy = batch[1]
        output, mask = model(real_noisy)
        print(output.shape)
        break


