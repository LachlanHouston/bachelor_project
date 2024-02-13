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
                    discriminator = Discriminator(),
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

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, p=1):
        # Compute the Lp loss between the real clean and the fake clean
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # Normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / fake_clean.numel()
        # compute adversarial loss
        G_adv_loss = - D_fake.mean()
        # Compute the total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss
        G_loss /= self.n_critic
        return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss
    
    def _get_discriminator_loss(self, real_clean, fake_clean, D_real, D_fake_no_grad):
        # compute gradient penalty
        alpha = torch.rand(real_clean.size(0), 1, 1, 1, device=self.device)
        difference = fake_clean - real_clean
        interpolates = real_clean + (alpha * difference)
        D_interpolate = self.discriminator(interpolates)
        grad_outputs = torch.ones(D_interpolate.size(), device=self.device)
        gradients = torch.autograd.grad(outputs=D_interpolate, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=(1, 2, 3)))
        gradient_penalty = torch.mean((slopes - 1.) ** 2)
        # compute the adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()
        D_loss = self.alpha_penalty * gradient_penalty + D_adv_loss
        return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)#, betas = (0., 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)#, betas = (0., 0.9))
        g_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=10, gamma=0.1)
        d_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_opt, step_size=10, gamma=0.1)
        return [g_opt, d_opt], [g_lr_scheduler, d_lr_scheduler]

    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        g_sch, d_sch = self.lr_schedulers()
        d_opt.zero_grad()
        if batch_idx % self.n_critic == 0 and batch_idx > 0:
            g_opt.zero_grad()

        real_clean = torch.stack(batch[0], dim=1).squeeze(0)
        real_noisy = torch.stack(batch[1], dim=1).squeeze(0)
        # real_clean = torch.randn(2, 2, 257, 321, device=self.device) # dummy variables for testing
        # real_noisy = torch.randn(2, 2, 257, 321, device=self.device)

        fake_clean, mask = self.generator(real_noisy)

        D_real = self.discriminator(real_clean)
        D_fake = self.discriminator(fake_clean)
        # detach fake_clean to avoid computing gradients for the generator when computing discriminator loss
        D_fake_no_grad = self.discriminator(fake_clean.detach())

        # detach fake_clean to avoid computing gradients for the generator
        D_loss, D_gp_alpha, D_adv_loss = self._get_discriminator_loss(real_clean=real_clean, fake_clean=fake_clean, D_real=D_real, D_fake_no_grad=D_fake_no_grad)
        G_loss, G_fidelity_alpha, G_adv_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, D_fake=D_fake)

        self.manual_backward(D_loss, retain_graph=True)
        self.manual_backward(G_loss)

        # Gradient clipping
        # self.clip_gradients(d_opt, gradient_clip_val=0.5, gradient_clip_algorithm='norm')   

        d_opt.step()

        if batch_idx % self.n_critic == 0 and batch_idx > 0:
            g_opt.step()

        # Weight clipping
        for p in self.discriminator.parameters():
            clip_value = 0.01
            p.data.clamp_(-clip_value, clip_value)
            
        # Update learning rate every epoch
        if self.trainer.is_last_batch and (self.trainer.current_epoch + 1) % 1 == 0:
            g_sch.step()
            d_sch.step()

        # Distance between real clean and fake clean
        # dist = torch.norm(real_clean - fake_clean, p=1)

        if self.visualize:
            if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
                visualize_stft_spectrogram(fake_clean[0], use_wandb = True)

                fake_clean_waveform = stft_to_waveform(fake_clean[0], device=self.device)
                waveform_np = fake_clean_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Generated Clean Audio")]})

                mask_waveform = stft_to_waveform(mask[0], device=self.device)
                waveform_np = mask_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"mask_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Learned Mask by Generator")]})
                
                real_noisy_waveform = stft_to_waveform(real_noisy[0], device=self.device)
                waveform_np = real_noisy_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Original Noisy Audio")]})
 
                real_clean_waveform = stft_to_waveform(real_clean[0], device=self.device)
                waveform_np = real_clean_waveform.detach().cpu().numpy().squeeze()
                self.logger.experiment.log({"real_clean_waveform": [wandb.Audio(waveform_np, sample_rate=16000, caption="Original Clean Audio")]})

            # log discriminator losses
            self.log('D_loss', D_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_real', D_real.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_fake', D_fake.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_Penalty', D_gp_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            # log generator losses
            self.log('G_loss', G_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_adv', G_adv_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) # opposite sign as D_fake
            self.log('G_Fidelity', G_fidelity_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            # self.log('Distance (true clean and fake clean)', dist, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # Compute batch SNR
        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        fake_clean = self.generator(real_noisy)

        # Signal to Noise Ratio (needs real_clean fake_clean to be paired)
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
    # train_loader, val_loader, test_loader = data_loader('data/clean_processed/', 'data/noisy_processed/', batch_size=1, num_workers=8)
    # # print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    # Dummy train_loader
    train_loader = torch.utils.data.DataLoader(
        torch.randn(2, 2, 257, 321),
        batch_size=2,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torch.randn(16, 2, 257, 321),
        batch_size=16,
        shuffle=True
    )

    test_loader = torch.utils.data.DataLoader(
        torch.randn(16, 2, 257, 321),
        batch_size=16,
        shuffle=True
    )

    model = Autoencoder(discriminator=Discriminator(), generator=Generator())
    trainer = L.Trainer(max_epochs=2, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=2, limit_val_batches=3, limit_test_batches=1,
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

