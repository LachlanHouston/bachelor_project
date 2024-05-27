from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.utils.utils import stft_to_waveform, perfect_shuffle, visualize_stft_spectrogram
import pytorch_lightning as L
from torch.optim import Adam
import torch
import numpy as np
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
import wandb


# Define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        # Define the generator
        self.generator = Generator(in_channels=2, out_channels=2).to(self.device)
        self.custom_global_step = 0
        self.save_hyperparameters(kwargs)  # Save hyperparameters to Weights and Biases
        self.automatic_optimization = False

    def forward(self, real_noisy):
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, real_clean, p=1):
        # Compute the Lp loss between the real noisy and the fake clean
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # Normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))

        # Compute SI-SDR loss
        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()
        objective_model = SQUIM_OBJECTIVE.get_model()
        _, _, si_sdr_pred = objective_model(fake_clean_waveforms)
        # Define the loss as the negative mean of the SI-SDR
        sisnr_loss = -si_sdr_pred.mean()
        sisnr_loss *= self.sisnr_loss

        # Total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + sisnr_loss
        return G_loss, self.alpha_fidelity * G_fidelity_loss, sisnr_loss

    def configure_optimizers(self):
        g_opt = Adam(self.generator.parameters(), lr=self.g_learning_rate)
        return g_opt

    def training_step(self, batch, batch_idx):
        g_opt = self.optimizers()
        # Unpack batched data
        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)

        # Generate fake clean
        fake_clean, mask = self.generator(real_noisy)
        G_loss, G_fidelity_alpha, sisnr_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, real_clean=real_clean)
        
        # Compute generator gradients
        self.manual_backward(G_loss)
        g_opt.step()
        g_opt.zero_grad()

        # Log generator losses
        self.log('G_Loss', G_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('G_Fidelity', G_fidelity_alpha, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('G_SI-SDR_Loss', sisnr_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.custom_global_step % 10 == 0 and self.dataset == "AudioSet":
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean.detach(), device=self.device).cpu().squeeze()
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SDR training', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.log_all_scores and self.custom_global_step % 50 == 0:
            fake_clean_waveforms = stft_to_waveform(fake_clean.detach(), device=self.device).cpu().squeeze()
            # Predicted objective metric: SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            _, _, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('SI-SDR pred Training', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        self.custom_global_step += 1


    def on_train_epoch_end(self):
        # log the norms of the generator and discriminator parameters
        if self.current_epoch % 1 == 0:
            for name, param in self.generator.named_parameters():
                self.log(f'Weight Norms/Gen_{name}_norm', param.norm(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            # also log the overall norm of the generator and discriminator parameters
            self.log('Weight Norms/Gen_mean_norm', torch.norm(torch.cat([param.view(-1) for param in self.generator.parameters()])), on_step=False, on_epoch=True, prog_bar=False, logger=True)


    def validation_step(self, batch, batch_idx):
        # remove tuples and convert to tensors
        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)

        fake_clean, mask = self.generator(real_noisy)

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()

        if self.dataset != 'AudioSet':
            # Scale Invariant Signal-to-Noise Ratio
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SNR', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Extended Short Time Objective Intelligibility
            estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
            estoi_score = estoi(preds = fake_clean_waveforms, target = real_clean_waveforms)
            self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # if self.current_epoch % 10 == 0 and batch_idx % 5 == 0:
        # Mean Opinion Score (SQUIM)
        reference_waveforms = perfect_shuffle(real_clean_waveforms)
        subjective_model = SQUIM_SUBJECTIVE.get_model()
        mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
        self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if (self.log_all_scores or self.dataset == "AudioSet") and batch_idx % 10 == 0:
            # Predicted objective metrics: STOI, PESQ, and SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('SI-SDR Pred', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI Pred', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # visualize the waveforms and spectrograms
        if batch_idx == 0:
            self.vis_batch_idx = torch.randint(0, (int(824*self.val_fraction)) // self.batch_size, (1,)).item() if self.dataset != 'dummy' else 0
        if batch_idx == self.vis_batch_idx:
            vis_idx = torch.randint(0, self.batch_size, (1,)).item() if self.dataset != 'dummy' else 0
            # log waveforms
            fake_clean_waveform = stft_to_waveform(fake_clean[vis_idx], device=self.device).cpu().numpy().squeeze()
            mask_waveform =       stft_to_waveform(mask[vis_idx],       device=self.device).cpu().numpy().squeeze()
            real_noisy_waveform = stft_to_waveform(real_noisy[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_clean_waveform = stft_to_waveform(real_clean[vis_idx], device=self.device).cpu().numpy().squeeze()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(fake_clean_waveform, sample_rate=16000, caption="Generated Clean Audio")]})
            self.logger.experiment.log({"mask_waveform":       [wandb.Audio(mask_waveform,       sample_rate=16000, caption="Learned Mask by Generator")]})
            self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(real_noisy_waveform, sample_rate=16000, caption="Original Noisy Audio")]})
            self.logger.experiment.log({"real_clean_waveform": [wandb.Audio(real_clean_waveform, sample_rate=16000, caption="Original Clean Audio")]})
            # log spectrograms
            plt = visualize_stft_spectrogram(real_clean_waveform, fake_clean_waveform, real_noisy_waveform)
            self.logger.experiment.log({"Spectrogram": [wandb.Image(plt, caption="Spectrogram")]})
            plt.close()
            plt = visualize_stft_spectrogram(mask_waveform, np.zeros_like(mask_waveform), np.zeros_like(mask_waveform))
            self.logger.experiment.log({"Mask": [wandb.Image(plt, caption="Mask")]})
            plt.close()


if __name__ == "__main__":
    # pytorch lightning trainer with dummy data loaders for testing
    train_loader = torch.utils.data.DataLoader([torch.randn(4, 2, 257, 321), torch.randn(4, 2, 257, 321)], batch_size=4, shuffle=True)
    val_loader = torch.utils.data.DataLoader([torch.randn(4, 2, 257, 321), torch.randn(4, 2, 257, 321)], batch_size=4, shuffle=False)
    
    model = Autoencoder(discriminator=Discriminator(), generator=Generator(), alpha_penalty=10, alpha_fidelity=10,
                        n_critic=1, d_learning_rate=1e-4, g_learning_rate=1e-4,
                        batch_size=4, log_all_scores=True, val_fraction = 1.)
    
    trainer = L.Trainer(max_epochs=5, accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=20, limit_val_batches=0,logger=False, fast_dev_run=False)
    
    trainer.fit(model, train_loader, val_loader)
