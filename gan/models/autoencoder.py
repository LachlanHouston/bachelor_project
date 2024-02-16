from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.data.data_loader import data_loader
import pytorch_lightning as L
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from matplotlib import pyplot as plt
import numpy as np
import io
import wandb


def visualize_stft_spectrogram(real_clean, fake_clean, real_noisy, use_wandb = False):
    """
    Visualizes an STFT-transformed file as a spectrogram and returns the plot as an image object
    for logging to wandb.
    
    Parameters:
    - stft_data: np.ndarrays with shape (2, Frequency bins, Frames). The first dimension contains
                 the real and imaginary parts of the STFT.
    
    Returns:
    - A BytesIO object containing the image of the plot.
    """
    
    complex_stft_rc = real_clean[0] + 1j * real_clean[1]
    complex_stft_rc = complex_stft_rc.cpu()
    magnitude_spectrum_rc = np.abs(complex_stft_rc.detach().numpy())

    complex_stft_fc = fake_clean[0] + 1j * fake_clean[1]
    complex_stft_fc = complex_stft_fc.cpu()
    magnitude_spectrum_fc = np.abs(complex_stft_fc.detach().numpy())

    complex_stft_rn = real_noisy[0] + 1j * real_noisy[1]
    complex_stft_rn = complex_stft_rn.cpu()
    magnitude_spectrum_rn = np.abs(complex_stft_rn.detach().numpy())

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    axs[0].imshow(magnitude_spectrum_rc, aspect='auto', origin='lower',
                    extent=[0, magnitude_spectrum_rc.shape[1], 0, magnitude_spectrum_rc.shape[0]])
    axs[0].set_title('Real Clean')
    axs[0].set_xlabel('Time (Frames)')
    axs[0].set_ylabel('Frequency (Bins)')
    axs[0].set_xticks(np.arange(0, magnitude_spectrum_rc.shape[1], 50))
    axs[0].set_yticks(np.arange(0, magnitude_spectrum_rc.shape[0], 50))

    axs[1].imshow(magnitude_spectrum_fc, aspect='auto', origin='lower',
                    extent=[0, magnitude_spectrum_fc.shape[1], 0, magnitude_spectrum_fc.shape[0]])
    axs[1].set_title('Fake Clean')
    axs[1].set_xlabel('Time (Frames)')
    axs[1].set_ylabel('Frequency (Bins)')
    axs[1].set_xticks(np.arange(0, magnitude_spectrum_fc.shape[1], 50))
    axs[1].set_yticks(np.arange(0, magnitude_spectrum_fc.shape[0], 50))

    axs[2].imshow(magnitude_spectrum_rn, aspect='auto', origin='lower',
                    extent=[0, magnitude_spectrum_rn.shape[1], 0, magnitude_spectrum_rn.shape[0]])
    axs[2].set_title('Real Noisy')
    axs[2].set_xlabel('Time (Frames)')
    axs[2].set_ylabel('Frequency (Bins)')
    axs[2].set_xticks(np.arange(0, magnitude_spectrum_rn.shape[1], 50))
    axs[2].set_yticks(np.arange(0, magnitude_spectrum_rn.shape[0], 50))

    fig.suptitle('Spectrograms')
    plt.tight_layout(pad=3.0)

    # # Create figure without displaying it
    # plt.figure(figsize=(10, 6))
    # plt.imshow(magnitude_spectrum_rc, aspect='auto', origin='lower', 
    #            extent=[0, magnitude_spectrum_rc.shape[1], 0, magnitude_spectrum_rc.shape[0]])
    # plt.colorbar(label='Magnitude')
    # plt.xlabel('Time (Frames)')
    # plt.ylabel('Frequency (Bins)')
    # plt.title('Amplitude Spectrogram')
    
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
                    d_scheduler_step_size=200,
                    d_scheduler_gamma=0.5,
                    g_scheduler_step_size=200,
                    g_scheduler_gamma=0.5,
                    weight_clip = False,
                    weight_clip_value = 0.01,
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
        self.d_scheduler_step_size = d_scheduler_step_size
        self.d_scheduler_gamma = d_scheduler_gamma
        self.g_scheduler_step_size = g_scheduler_step_size
        self.g_scheduler_gamma = g_scheduler_gamma
        self.weight_clip = weight_clip
        self.weight_clip_value = weight_clip_value
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
        g_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=self.g_scheduler_step_size, gamma=self.g_scheduler_gamma)
        d_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_opt, step_size=self.d_scheduler_step_size, gamma=self.d_scheduler_gamma)
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

        d_opt.step()

        if batch_idx % self.n_critic == 0 and batch_idx > 0:
            g_opt.step()

        # Weight clipping
        if self.weight_clip:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)
            
        # Update learning rate every epoch
        if self.trainer.is_last_batch:
            g_sch.step()
            d_sch.step()

        if self.visualize:
            # log discriminator losses
            self.log('D_Loss', D_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_Real', D_real.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_Fake', D_fake.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('D_Penalty', D_gp_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            
            # log generator losses
            self.log('G_Loss', G_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Adversarial', G_adv_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) # opposite sign as D_fake
            self.log('G_Fidelity', G_fidelity_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)

    def validation_step(self, batch, batch_idx):
        # Remove tuples and convert to tensors
        real_clean = torch.stack(batch[0], dim=1).squeeze(0)
        real_noisy = torch.stack(batch[1], dim=1).squeeze(0)

        fake_clean, mask = self.generator(real_noisy)

        # Scale Invariant Signal to Noise Ratio
        snr = ScaleInvariantSignalNoiseRatio().to(self.device)
        snr_val = snr(real_clean, fake_clean)
        self.log('SI-SNR (test set)', snr_val, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Perceptual Evaluation of Speech Quality
        # real_clean_waveform = stft_to_waveform(real_clean[0], device=self.device)
        # real_clean_waveform = real_clean_waveform.detach().cpu().squeeze()
        # fake_clean_waveform = stft_to_waveform(fake_clean[0], device=self.device)
        # fake_clean_waveform = fake_clean_waveform.detach().cpu().squeeze()

        # pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(self.device)
        # pesq_val = pesq(real_clean_waveform, fake_clean_waveform)
        # self.log('val_PESQ', pesq_val, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Distance between real clean and fake clean
        dist = torch.norm(real_clean - fake_clean, p=1)
        self.log('Distance - true clean and fake clean (test set)', dist, on_step=True, on_epoch=False, prog_bar=True, logger=True)


        if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
            
            vis_idx = torch.randint(0, real_clean.shape[0], (1,)).item()

            visualize_stft_spectrogram(real_clean[vis_idx], fake_clean[vis_idx], real_noisy[vis_idx], use_wandb = True)

            fake_clean_waveform = stft_to_waveform(fake_clean[vis_idx], device=self.device)
            fake_clean_waveform = fake_clean_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(fake_clean_waveform, sample_rate=16000, caption="Generated Clean Audio")]})

            mask_waveform = stft_to_waveform(mask[vis_idx], device=self.device)
            mask_waveform = mask_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"mask_waveform": [wandb.Audio(mask_waveform, sample_rate=16000, caption="Learned Mask by Generator")]})
            
            real_noisy_waveform = stft_to_waveform(real_noisy[vis_idx], device=self.device)
            real_noisy_waveform = real_noisy_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(real_noisy_waveform, sample_rate=16000, caption="Original Noisy Audio")]})

            real_clean_waveform = stft_to_waveform(real_clean[vis_idx], device=self.device)
            real_clean_waveform = real_clean_waveform.detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"real_clean_waveform": [wandb.Audio(real_clean_waveform, sample_rate=16000, caption="Original Clean Audio")]})


if __name__ == "__main__":
    # Print Device
    print(torch.cuda.is_available())
    train_loader, val_loader = data_loader('data/clean_processed/', 'data/noisy_processed/', 
                                           'data/test_clean_processed/', 'data/test_noisy_processed/',
                                           batch_size=1, num_workers=8)
    # # print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    # Dummy train_loader
    # train_loader = torch.utils.data.DataLoader(
    #     torch.randn(2, 2, 257, 321),
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
    trainer = L.Trainer(max_epochs=1, accelerator='auto', num_sanity_val_steps=2,
                        log_every_n_steps=1, limit_train_batches=2, limit_val_batches=3, limit_test_batches=1,
                        logger=False)
    trainer.fit(model, train_loader, val_loader)
