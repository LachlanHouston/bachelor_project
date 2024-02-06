from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
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

def SI_SNR(target, estimate, eps=1e-8):
    target = torch.stack(target, dim=1).squeeze(0)
    estimate = torch.stack(estimate, dim=1).squeeze(0)

    target = target - torch.mean(target, -1, keepdim=True)
    estimate = estimate - torch.mean(estimate, -1, keepdim=True)

    s1 = torch.sum(target * estimate, -1, keepdim=True)
    s2 = torch.sum(estimate * estimate, -1, keepdim=True)

    s_target = s1 * estimate / (s2 + eps) * estimate
    e_noise = target - s_target

    target_norm = torch.sum(s_target * s_target, -1, keepdim=True)
    noise_norm = torch.sum(e_noise * e_noise, -1, keepdim=True)

    snr = 10 * torch.log10(target_norm / (noise_norm + eps) + eps)
    return torch.mean(snr)





class Autoencoder(L.LightningModule):
    def __init__(self, 
                    discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128]),
                    generator = Generator(),
                    alpha_penalty=10,
                    alpha_fidelity=10,
                    n_critic=10,
                    logging_freq=5
                 ):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])
        self.alpha_penalty = alpha_penalty
        self.alpha_fidelity = alpha_fidelity
        self.n_critic = n_critic
        self.logging_freq = logging_freq

        self.automatic_optimization = False



    def forward(self, real_noisy):
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, d_fake, fake_clean, real_noisy, p=1):
        G_adv_loss = torch.mean(d_fake)
        fake_clean_cat = torch.cat((fake_clean, fake_clean), dim=1)
        real_noisy_cat = torch.cat((real_noisy, real_noisy), dim=1)
        G_fidelity_loss = torch.norm(fake_clean_cat - real_noisy_cat, p=p)**p

        G_loss = self.alpha_fidelity * G_fidelity_loss - G_adv_loss
        return G_loss
    
    def _get_discriminator_loss(self, d_real, d_fake, real_input, fake_input):
        alpha = torch.rand(real_input.size(0), 1, 1, 1, device=self.device)

        difference = fake_input - real_input
        interpolates = real_input + alpha * difference
        
        out = self.discriminator(interpolates)
        grad_outputs = torch.ones(out.size(), device=self.device)

        gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        slopes = gradients.view(gradients.size(0), -1).norm(2, dim=1)
        gradient_penalty = torch.mean((slopes - 1.) ** 2)

        D_adv_loss = d_fake.mean() - d_real.mean()
        D_loss = D_adv_loss + self.alpha_penalty * gradient_penalty

        return D_loss
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=1e-4)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=1e-4)

        return [g_opt, d_opt], []

    def training_step(self, batch, batch_idx):
        #torch.autograd.set_detect_anomaly(True)
        g_opt, d_opt = self.optimizers()
        

        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        d_real = self.discriminator(real_clean)
        fake_clean = self.generator(real_noisy)
        d_fake = self.discriminator(fake_clean)

        # Train the generator
        G_loss = self._get_reconstruction_loss(d_fake, fake_clean, real_noisy, p=1)

        # Train the discriminator
        D_loss = self._get_discriminator_loss(d_real, d_fake, real_clean, fake_clean)

        g_opt.zero_grad()
        d_opt.zero_grad()

        self.manual_backward(G_loss, retain_graph=True)
            

        self.manual_backward(D_loss)

        d_opt.step()

        if batch_idx % self.n_critic == 0 and batch_idx > 0:
            g_opt.step()

        distance = torch.norm(real_noisy - fake_clean, p=1)
        
        if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
            visualize_stft_spectrogram(fake_clean[0], use_wandb = True)

            fake_clean_waveform = stft_to_waveform(fake_clean[0], device=self.device)
            waveform_np = fake_clean_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Generated Clean Audio")]})
            
            real_noisy_waveform = stft_to_waveform(real_noisy[0], device=self.device)
            waveform_np = real_noisy_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Original Noisy Audio")]})

        self.log('D_loss', D_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('G_loss', G_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('distance', distance, on_step=True, on_epoch=True, prog_bar=True, logger=True)

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
    train_loader, val_loader, test_loader = data_loader('data/clean_processed/', 'data/noisy_processed/', batch_size=16, num_workers=8)
    print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    model = Autoencoder(discriminator=Discriminator(), generator=Generator())
    trainer = L.Trainer(max_epochs=2, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=12, limit_val_batches=3, limit_test_batches=1,
                        logger=False)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard

    # Create dummy data of one batch
    # batch = next(iter(train_loader))
    # real_clean = batch[0]
    # real_noisy = batch[1]
    
    # SNR = SI_SNR(real_clean, real_noisy)
    # print(SNR)

