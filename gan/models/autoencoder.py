from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.data.data_loader import data_loader
import pytorch_lightning as L
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from matplotlib import pyplot as plt
import numpy as np
import io
import wandb
from scipy.io import wavfile


def visualize_stft_spectrogram(stft_data, use_wandb = False):
    """
    Visualizes an STFT-transformed file as a spectrogram and returns the plot as an image object
    for logging to wandb.
    
    Parameters:
    - stft_data: np.ndarray with shape (2, Frequency bins, Frames). The first dimension contains
                 the real and imaginary parts of the STFT.
    
    Returns:
    - A BytesIO object containing the image of the plot.
    """
    if stft_data.shape[0] != 2:
        raise ValueError("stft_data should have a shape (2, Frequency bins, Frames)")
    
    complex_stft = stft_data[0] + 1j * stft_data[1]
    complex_stft = complex_stft.cpu()
    magnitude_spectrum = np.abs(complex_stft.detach().numpy())
    
    
    # Create figure without displaying it
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_spectrum, aspect='auto', origin='lower', 
               extent=[0, magnitude_spectrum.shape[1], 0, magnitude_spectrum.shape[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Frequency (Bins)')
    plt.title('Amplitude Spectrogram')
    
    if use_wandb:
        image =  wandb.Image(plt)
        wandb.log({"Spectrogram": wandb.Image(plt)})
        # Create a bytes buffer for the image to avoid saving to disk
        buf = io.BytesIO()
        # Save the plot to the buffer
        plt.savefig(buf, format='png')
        # Important: Close the plot to free memory
        plt.close()
        
        # Reset buffer's cursor to the beginning
        buf.seek(0)
        # image = Image.open(buf)
        # return image
        return buf
    else:
        plt.show()

def stft_to_waveform(stft, device=torch.device('cuda')):
    if len(stft.shape) == 3:
        stft = stft.unsqueeze(0)
    # Separate the real and imaginary components
    stft_real = stft[:, 0, :, :]
    stft_imag = stft[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)
    # Perform inverse STFT to obtain the waveform
    waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device))
    return waveform

class Autoencoder(L.LightningModule):
    def __init__(self, 
                    discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128]),
                    generator = Generator(),
                    alpha_penalty=10,
                    alpha_fidelity=10,
                    n_critic=5,
                    n_generator=1,
                    logging_freq=5,
                    d_learning_rate=1e-4,
                    g_learning_rate=1e-4,
                    visualize=False
                 ):
        super().__init__()
        self.generator = generator
        self.discriminator = discriminator
        self.alpha_penalty = alpha_penalty
        self.alpha_fidelity = alpha_fidelity
        self.n_critic = n_critic
        self.n_generator = n_generator
        self.logging_freq = logging_freq
        self.d_learning_rate = d_learning_rate
        self.g_learning_rate = g_learning_rate
        self.visualize = visualize

        self.automatic_optimization = False
        self.save_hyperparameters()

    def forward(self, real_noisy):
        return self.generator(real_noisy)
    
    # def get_infinite_dataloader(self, batch):
    #     while True:
    #         for data in batch:
    #             yield data

    def _get_reconstruction_loss(self, fake_clean, real_noisy, p=1):
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)

        G_fidelity_loss = G_fidelity_loss / fake_clean.numel()
        G_loss = self.alpha_fidelity * G_fidelity_loss
        return G_loss
    
    def _get_discriminator_loss(self, real_input, fake_input):
        alpha = torch.rand(real_input.size(0), 1, 1, 1, device=self.device)

        interpolates = alpha * real_input + (1 - alpha) * fake_input
        
        out = self.discriminator(interpolates)
        grad_outputs = torch.ones(out.size(), device=self.device)

        gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        slopes = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        gradient_penalty = torch.mean((slopes - 1.) ** 2)

        D_loss = self.alpha_penalty * gradient_penalty

        return D_loss
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)
        return [g_opt, d_opt], []

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        real_clean = batch[0]
        real_noisy = batch[1]

        # real_clean = torch.randn(16, 2, 257, 321, device=self.device)
        # real_noisy = torch.randn(16, 2, 257, 321, device=self.device)

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        fake_clean = self.generator(real_noisy)

        d_real = self.discriminator(real_clean)
        d_fake = self.discriminator(fake_clean)

        disc_cost = d_fake.mean() - d_real.mean()
        gen_cost = -d_fake.mean()

        gradient_penalty = self._get_discriminator_loss(real_clean, fake_clean)
        fidelity = self._get_reconstruction_loss(fake_clean, real_noisy)

        t_disc_cost = disc_cost + gradient_penalty
        t_gen_cost = (gen_cost + fidelity) / self.n_critic
        
        d_opt.zero_grad()

        self.manual_backward(t_disc_cost, retain_graph=True)
        self.manual_backward(t_gen_cost)
        
        if batch_idx % self.n_critic == 0 and batch_idx > 0:
            g_opt.step()
            g_opt.zero_grad()

        d_opt.step()

        # Distance between real clean and fake clean
        dist = torch.norm(real_clean - fake_clean, p=1)

        if self.visualize:
            if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
                visualize_stft_spectrogram(fake_clean[0], use_wandb = True)

                fake_clean_waveform = stft_to_waveform(fake_clean[0], device=self.device)
                waveform_np = fake_clean_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Generated Clean Audio")]})
                
                real_noisy_waveform = stft_to_waveform(real_noisy[0], device=self.device)
                waveform_np = real_noisy_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Original Noisy Audio")]})

            self.log('D_loss', t_disc_cost, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_loss', t_gen_cost, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_real', d_real.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_fake', d_fake.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_adv', gen_cost, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('Penalty', gradient_penalty, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('Fidelity', fidelity, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('Distance (true clean and fake clean)', dist, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # Compute batch SNR
        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        fake_clean = self.generator(real_noisy)

        # Signal to Noise Ratio
        snr = ScaleInvariantSignalNoiseRatio().to(self.device)
        snr_val = snr(real_clean, fake_clean)

        self.log('val_SNR', snr_val, on_step=True, on_epoch=True, prog_bar=True, logger=True)

    def test_step(self, batch, batch_idx):
        # Compute test SNR
        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        fake_clean = self.generator(real_noisy)

        # Signal to Noise Ratio
        snr = ScaleInvariantSignalNoiseRatio().to(self.device)
        snr_val = snr(real_clean, fake_clean)

        self.log('test_SNR', snr_val)

if __name__ == "__main__":
    # Print Device
    print(torch.cuda.is_available())
    train_loader, val_loader, test_loader = data_loader('data/clean_processed/', 'data/noisy_processed/', batch_size=1, num_workers=8)
    # # print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    # Dummy train_loader
    # train_loader = torch.utils.data.DataLoader(
    #     torch.randn(16, 2, 257, 321),
    #     batch_size=2,
    #     shuffle=True
    # )

    # val_loader = torch.utils.data.DataLoader(
    #     torch.randn(16, 2, 257, 321),
    #     batch_size=16,
    #     shuffle=True
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     torch.randn(16, 2, 257, 321),
    #     batch_size=16,
    #     shuffle=True
    # )

    model = Autoencoder(discriminator=Discriminator(), generator=Generator())
    trainer = L.Trainer(max_epochs=5, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=12, limit_val_batches=3, limit_test_batches=1,
                        logger=False)
    trainer.fit(model, train_loader)
    # trainer.test(model, test_loader)

    # trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard

    # Create dummy data of one batch
    # batch = next(iter(train_loader))
    # real_clean = batch[0]
    # real_noisy = batch[1]
    
    # SNR = SI_SNR(real_clean, real_noisy)
    # print(SNR)

