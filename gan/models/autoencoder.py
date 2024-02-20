from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.data.data_loader import VCTKDataModule
import pytorch_lightning as L
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
# from torchmetrics.audio import PerceptualEvaluationSpeechQuality
# from gan.utils.utils import SegSNR
from speechmos import dnsmos
from matplotlib import pyplot as plt
import numpy as np
import io
import wandb
import librosa
import librosa.display
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True

def visualize_stft_spectrogram(real_clean, fake_clean, real_noisy, use_wandb = False):
    """
    Visualizes a STFT-transformed files as mel spectrograms and returns the plot as an image object
    for logging to wandb.
    """    

    S_real_c = real_clean[0].cpu()
    S_fake_c = fake_clean[0].cpu()
    S_real_n = real_noisy[0].cpu()

    # Spectrogram of real clean
    mel_spect_rc = librosa.feature.melspectrogram(S=S_real_c, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_rc = librosa.power_to_db(mel_spect_rc, ref=np.max)
    # Spectrogram of fake clean
    mel_spect_fc = librosa.feature.melspectrogram(S=S_fake_c, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_fc = librosa.power_to_db(mel_spect_fc, ref=np.max)
    # Spectrogram of real noisy
    mel_spect_rn = librosa.feature.melspectrogram(S=S_real_n, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_rn = librosa.power_to_db(mel_spect_rn, ref=np.max)
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Define Real Clean plot
    img_rc = librosa.display.specshow(mel_spect_db_rc, ax=axs[0], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_rc, ax=axs[0], format='%+2.0f dB')
    axs[0].set_title('Real Clean')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency (Hz)')

    # Define Fake Clean plot
    img_fc = librosa.display.specshow(mel_spect_db_fc, ax=axs[1], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_fc, ax=axs[1], format='%+2.0f dB')
    axs[1].set_title('Fake Clean')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')

    # Define Real Noisy plot
    img_rn = librosa.display.specshow(mel_spect_db_rn, ax=axs[2], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_rn, ax=axs[2], format='%+2.0f dB')
    axs[2].set_title('Real Noisy')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Frequency (Hz)')

    # Set the title of the figure
    fig.suptitle('Spectrograms')
    plt.tight_layout(pad=3.0)
    
    if use_wandb:
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
        # Enable gradient calculation for interpolates
        interpolates.requires_grad_(True)

        D_interpolate = self.discriminator(interpolates)
        ones = torch.ones(D_interpolate.size(), device=self.device)
        gradients = torch.autograd.grad(outputs=D_interpolate, inputs=interpolates, grad_outputs=ones, create_graph=True, retain_graph=True, only_inputs=True)[0]

        # slopes = torch.sqrt(torch.sum(gradients ** 2, dim=(1, 2, 3)))
        slopes = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-10) # Adding a small epsilon to prevent division by zero

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

        real_clean = batch[0].squeeze(1)
        real_noisy = batch[1].squeeze(1)

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
        real_clean = batch[0].squeeze(1)
        real_noisy = batch[1].squeeze(1)

        fake_clean, mask = self.generator(real_noisy)

        ## Scale Invariant Signal-to-Noise Ratio
        snr = ScaleInvariantSignalNoiseRatio().to(self.device)
        snr_score = snr(real_clean, fake_clean)
        self.log('SI-SNR', snr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        real_clean_waveform = stft_to_waveform(real_clean, device=self.device)
        real_clean_waveform = real_clean_waveform.detach().cpu().squeeze()
        fake_clean_waveform = stft_to_waveform(fake_clean, device=self.device)
        fake_clean_waveform = fake_clean_waveform.detach().cpu().squeeze()
        
        ## Perceptual Evaluation of Speech Quality
        # pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(self.device)
        # pesq_score = pesq(real_clean_waveform, fake_clean_waveform)

        ## Deep Noise Suppression Mean Opinion Score (DNSMOS)
        dnsmos_score = np.mean([dnsmos.run(fake_clean_waveform.numpy()[i], 16000)['ovrl_mos'] for i in range(fake_clean_waveform.shape[0])])
        self.log('DNSMOS', dnsmos_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        ## Extended Short Time Objective Intelligibility
        estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
        estoi_score = estoi(preds = fake_clean_waveform, target = real_clean_waveform)
        self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ## Segmental Signal-to-Noise Ratio (SegSNR)
        # segsnr = SegSNR(seg_length=160) # 160 corresponds to 10ms of audio with sr=16000
        # segsnr.update(preds=fake_clean_waveform, target=real_clean_waveform)
        # segsnr_score = segsnr.compute() # Average SegSNR
        # self.log('SegSNR', segsnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # # Distance between real clean and fake clean
        # dist = torch.norm(real_clean - fake_clean, p=1)
        # self.log('Distance - real clean and fake clean', dist, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.visualize:
            if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
                visualize_stft_spectrogram(real_clean[0], fake_clean[0], real_noisy[0], use_wandb = True)

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
    train_loader = torch.utils.data.DataLoader(
        torch.randn(2, 1, 2, 257, 321),
        batch_size=2,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        torch.randn(2, 1, 2, 257, 321),
        batch_size=16,
        shuffle=True
    )

    model = Autoencoder(discriminator=Discriminator(), generator=Generator(), visualize=False)
    trainer = L.Trainer(max_epochs=5, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=5, limit_val_batches=1,
                        logger=False)
    trainer.fit(model, train_loader, val_loader)
