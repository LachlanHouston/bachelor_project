from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.utils.utils import stft_to_waveform, perfect_shuffle, visualize_stft_spectrogram
import pytorch_lightning as L
import torch
import numpy as np
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
import wandb
import torchaudio
# from pesq import pesq

# define the Autoencoder class containing the training setup
class AutoencoderMix(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.discriminator=Discriminator(use_bias=self.use_bias).to(self.device)
        self.generator=Generator(in_channels=2, out_channels=2).to(self.device)
        self.custom_global_step = 0
        # save hyperparameters to Weights and Biases
        self.save_hyperparameters(kwargs)
        self.automatic_optimization = False

    def forward(self, real_noisy):
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, real_clean, authentic=False, p=1):
        # Compute the Lp loss between fake clean and the original noisy signal
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # Normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))
        # compute adversarial loss
        G_adv_loss = - torch.mean(D_fake)

        # Compute the total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss

        if self.sisnr_loss and not authentic:
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_loss = - sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            sisnr_loss *= self.sisnr_loss
            G_loss += sisnr_loss
            return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss, sisnr_loss

        return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss, None
    
    def _get_discriminator_loss(self, real_clean, fake_clean, D_real, D_fake_no_grad):
        # Create interpolated samples
        alpha = torch.rand(int(self.batch_size/2), 1, 1, 1, device=self.device) # B x 1 x 1 x 1
        # alpha = alpha.expand(real_clean.size()) # B x C x H x W
        differences = fake_clean - real_clean # B x C x H x W
        interpolates = real_clean + (alpha * differences) # B x C x H x W
        interpolates.requires_grad_(True)

        # Calculate the output of the discriminator for the interpolated samples and compute the gradients
        D_interpolates = self.discriminator(interpolates) # B x 1 (the output of the discriminator is a scalar value for each input sample)
        ones = torch.ones(D_interpolates.size(), device=self.device) # B x 1
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W
        
        # Calculate the gradient penalty
        gradients = gradients.view(int(self.batch_size/2), -1) # B x (C*H*W)
        grad_norms = gradients.norm(2, dim=1) # B
        gradient_penalty = ((grad_norms - 1) ** 2).mean()

        # Adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()

        # Total discriminator loss
        D_loss = self.alpha_penalty * gradient_penalty + D_adv_loss

        if self.L2_reg:
            # L2 regularization on discriminator biases
            L2_penalty_bias = 0.0
            # Iterate over discriminator parameters and apply L2 regularization to bias terms
            for name, param in self.discriminator.named_parameters():
                if 'bias' in name:  # Check if the parameter is a bias
                    L2_penalty_bias += param.pow(2).sum()  # L2 penalty is the sum of squares of the parameter
            L2_penalty_bias *= self.L2_reg
            D_loss += L2_penalty_bias

            return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss, L2_penalty_bias

        return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss, None
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

        return [g_opt, d_opt], []
    
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()

        train_G = (self.custom_global_step + 1) % self.n_critic == 0
        paired_data, authentic_data = batch

        real_clean_paired = paired_data[0].squeeze(1).to(self.device)
        real_noisy_paired = paired_data[1].squeeze(1).to(self.device)
        real_clean_authentic = authentic_data[0].squeeze(1).to(self.device)
        real_noisy_authentic = authentic_data[1].squeeze(1).to(self.device)

        if train_G:
            self.toggle_optimizer(g_opt)
            # Generate fake clean for paired data
            fake_clean_paired, mask_paired = self.generator(real_noisy_paired)
            D_fake_paired = self.discriminator(fake_clean_paired)
            G_loss_paired, G_fidelity_alpha_paired, G_adv_loss_paired, sisnr_loss_paired = self._get_reconstruction_loss(real_noisy=real_noisy_paired, fake_clean=fake_clean_paired, D_fake=D_fake_paired, real_clean=real_clean_paired, authentic=False)
            # Generate fake clean for authentic data
            fake_clean_authentic, mask_authentic = self.generator(real_noisy_authentic)
            D_fake_authentic = self.discriminator(fake_clean_authentic)
            G_loss_authentic, G_fidelity_alpha_authentic, G_adv_loss_authentic, _ = self._get_reconstruction_loss(real_noisy=real_noisy_authentic, fake_clean=fake_clean_authentic, D_fake=D_fake_authentic, real_clean=None, authentic=True)
            # Compute generator gradients
            G_loss = G_loss_paired + G_loss_authentic
            self.manual_backward(G_loss)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        # Compute discriminator loss for paired data
        fake_clean_paired_no_grad = self.generator(real_noisy_paired)[0].detach()
        D_fake_paired_no_grad = self.discriminator(fake_clean_paired_no_grad)
        D_real_paired = self.discriminator(real_clean_paired)
        D_loss_paired, D_gp_alpha_paired, D_adv_loss_paired, L2_penalty_bias_paired = self._get_discriminator_loss(real_clean=real_clean_paired, fake_clean=fake_clean_paired_no_grad, D_real=D_real_paired, D_fake_no_grad=D_fake_paired_no_grad)
        # Compute discriminator loss for authentic data
        fake_clean_authentic_no_grad = self.generator(real_noisy_authentic)[0].detach()
        D_fake_authentic_no_grad = self.discriminator(fake_clean_authentic_no_grad)
        D_real_authentic = self.discriminator(real_clean_authentic)
        D_loss_authentic, D_gp_alpha_authentic, D_adv_loss_authentic, L2_penalty_bias_authentic = self._get_discriminator_loss(real_clean=real_clean_authentic, fake_clean=fake_clean_authentic_no_grad, D_real=D_real_authentic, D_fake_no_grad=D_fake_authentic_no_grad)
        # Compute total discriminator loss
        D_loss = D_loss_paired + D_loss_authentic
        # Compute discriminator gradients
        self.manual_backward(D_loss)
        # Update discriminator weights
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        # Weight clipping
        if self.weight_clip:
            for name, p in self.discriminator.named_parameters():
                # if 'bias' in name:               
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)

        # Compute combined losses
        D_loss_combined = D_loss_paired + D_loss_authentic
        D_fake_paired = D_fake_paired_no_grad
        D_fake_authentic = D_fake_authentic_no_grad

        # Log discriminator losses for paired, authentic, and combined datasets
        self.log('D_Loss_Paired', D_loss_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Loss_Authentic', D_loss_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Loss_Combined', D_loss_combined, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        # Log discriminator scores for real and fake samples, each dataset separately and combined
        self.log('D_Real_Paired', D_real_paired.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake_Paired', D_fake_paired.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Real_Authentic', D_real_authentic.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake_Authentic', D_fake_authentic.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        D_real_combined = torch.cat([D_real_paired, D_real_authentic]).mean()
        D_fake_combined = torch.cat([D_fake_paired, D_fake_authentic]).mean()
        self.log('D_Real_Combined', D_real_combined, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake_Combined', D_fake_combined, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Penalty_Paired', D_gp_alpha_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Penalty_Authentic', D_gp_alpha_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        # Log generator losses for paired, authentic, and combined datasets
        if train_G:
            self.log('G_Loss_Paired', G_loss_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Loss_Authentic', G_loss_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Loss_Combined', G_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Adversarial_Paired', G_adv_loss_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Adversarial_Authentic', G_adv_loss_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Adversarial_Combined', G_adv_loss_paired + G_adv_loss_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Fidelity_Paired', G_fidelity_alpha_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Fidelity_Authentic', G_fidelity_alpha_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Fidelity_Combined', G_fidelity_alpha_paired + G_fidelity_alpha_authentic, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            if self.sisnr_loss:
                self.log('G_SI-SNR_Loss_Paired', sisnr_loss_paired, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.custom_global_step % 10 == 0 and self.dataset == "VCTK":        
            real_clean_waveforms_paired = stft_to_waveform(real_clean_paired, device=self.device).cpu().squeeze()
            fake_clean_waveforms_paired = stft_to_waveform(fake_clean_paired_no_grad, device=self.device).cpu().squeeze()
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms_paired, target=real_clean_waveforms_paired)
            self.log('SI-SNR training', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.log_all_scores and self.custom_global_step % 50 == 0:
            fake_clean_waveforms_paired = stft_to_waveform(fake_clean_paired_no_grad, device=self.device).cpu().squeeze()
            ## Predicted objective metric: SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms_paired)
            self.log('SI-SDR pred Training Paired', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI pred Training Paired', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ pred Training Paired', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)

            fake_clean_waveforms_authentic = stft_to_waveform(fake_clean_authentic_no_grad, device=self.device).cpu().squeeze()
            ## Predicted objective metric: SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms_authentic)
            self.log('SI-SDR pred Training Authentic', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI pred Training Authentic', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ pred Training Authentic', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.custom_global_step += 1

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        paired_data, authentic_data = batch
        real_clean_paired = paired_data[0].squeeze(1).to(self.device)
        real_noisy_paired = paired_data[1].squeeze(1).to(self.device)
        real_clean_authentic = authentic_data[0].squeeze(1).to(self.device)
        real_noisy_authentic = authentic_data[1].squeeze(1).to(self.device)

        # Process paired data
        fake_clean_paired, mask_paired = self.generator(real_noisy_paired)
        real_clean_waveforms_paired = stft_to_waveform(real_clean_paired, device=self.device).cpu().squeeze()
        fake_clean_waveforms_paired = stft_to_waveform(fake_clean_paired, device=self.device).cpu().squeeze()

        # Process authentic data
        fake_clean_authentic, mask_authentic = self.generator(real_noisy_authentic)
        real_clean_waveforms_authentic = stft_to_waveform(real_clean_authentic, device=self.device).cpu().squeeze()
        fake_clean_waveforms_authentic = stft_to_waveform(fake_clean_authentic, device=self.device).cpu().squeeze()

        # SI-SNR scores
        if self.dataset == "VCTK":
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score_paired = sisnr(preds=fake_clean_waveforms_paired, target=real_clean_waveforms_paired)
            self.log('SI-SNR_Paired', sisnr_score_paired, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            # Extended Short Time Objective Intelligibility (eSTOI)
            estoi = ShortTimeObjectiveIntelligibility(16000, extended=True)
            estoi_score_paired = estoi(preds=fake_clean_waveforms_paired, target=real_clean_waveforms_paired)
            self.log('eSTOI_Paired', estoi_score_paired, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        # Mean Opinion Score (SQUIM) for paired, authentic, and combined
        if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            
            # For paired data
            reference_waveforms_paired = perfect_shuffle(real_clean_waveforms_paired)
            mos_squim_score_paired = torch.mean(subjective_model(fake_clean_waveforms_paired, reference_waveforms_paired)).item()
            self.log('MOS SQUIM_Paired', mos_squim_score_paired, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            # For authentic data
            reference_waveforms_authentic = perfect_shuffle(real_clean_waveforms_authentic)
            mos_squim_score_authentic = torch.mean(subjective_model(fake_clean_waveforms_authentic, reference_waveforms_authentic)).item()
            self.log('MOS SQUIM_Authentic', mos_squim_score_authentic, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            self.log('MOS SQUIM_Combined', np.mean([mos_squim_score_paired, mos_squim_score_authentic]), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Predicted objective metrics: STOI, PESQ, and SI-SDR for paired, authentic, and combined
        if (self.log_all_scores or self.dataset != "VCTK") and batch_idx % 50 == 0:
            objective_model = SQUIM_OBJECTIVE.get_model()
            
            # For paired data
            stoi_pred_paired, pesq_pred_paired, si_sdr_pred_paired = objective_model(fake_clean_waveforms_paired)
            self.log('SI-SDR Pred_Paired', si_sdr_pred_paired.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI Pred_Paired', stoi_pred_paired.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred_Paired', pesq_pred_paired.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # For authentic data
            stoi_pred_authentic, pesq_pred_authentic, si_sdr_pred_authentic = objective_model(fake_clean_waveforms_authentic)
            self.log('SI-SDR Pred_Authentic', si_sdr_pred_authentic.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI Pred_Authentic', stoi_pred_authentic.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred_Authentic', pesq_pred_authentic.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # Combined
            self.log('SI-SDR Pred_Combined', np.mean([si_sdr_pred_paired.mean(), si_sdr_pred_authentic.mean()]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI Pred_Combined', np.mean([stoi_pred_paired.mean(), stoi_pred_authentic.mean()]), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred_Combined', np.mean([pesq_pred_paired.mean(), pesq_pred_authentic.mean()]), on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # visualize the spectrogram and waveforms every first batch of every self.logging_freq epochs
        if batch_idx == 0:
            self.vis_batch_idx = torch.randint(0, (int(824*self.val_fraction)) // int(self.batch_size/2), (1,)).item()

        if batch_idx == self.vis_batch_idx and self.current_epoch % self.logging_freq == 0:
            vis_idx = torch.randint(0, int(self.batch_size/2), (1,)).item()

            # Paired data visualization
            fake_clean_waveform_paired = stft_to_waveform(fake_clean_paired[vis_idx], device=self.device).cpu().numpy().squeeze()
            mask_waveform_paired = stft_to_waveform(mask_paired[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_noisy_waveform_paired = stft_to_waveform(real_noisy_paired[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_clean_waveform_paired = stft_to_waveform(real_clean_paired[vis_idx], device=self.device).cpu().numpy().squeeze()

            self.logger.experiment.log({"fake_clean_waveform_paired": [wandb.Audio(fake_clean_waveform_paired, sample_rate=16000, caption="Generated Clean Audio - Paired")]})
            self.logger.experiment.log({"mask_waveform_paired": [wandb.Audio(mask_waveform_paired, sample_rate=16000, caption="Learned Mask by Generator - Paired")]})
            self.logger.experiment.log({"real_noisy_waveform_paired": [wandb.Audio(real_noisy_waveform_paired, sample_rate=16000, caption="Original Noisy Audio - Paired")]})
            self.logger.experiment.log({"real_clean_waveform_paired": [wandb.Audio(real_clean_waveform_paired, sample_rate=16000, caption="Original Clean Audio - Paired")]})

            plt = visualize_stft_spectrogram(real_clean_waveform_paired, fake_clean_waveform_paired, real_noisy_waveform_paired)
            self.logger.experiment.log({"Spectrogram_Paired": [wandb.Image(plt, caption="Spectrogram - Paired")]})
            plt.close()

            # Authentic data visualization
            fake_clean_waveform_authentic = stft_to_waveform(fake_clean_authentic[vis_idx], device=self.device).cpu().numpy().squeeze()
            mask_waveform_authentic = stft_to_waveform(mask_authentic[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_noisy_waveform_authentic = stft_to_waveform(real_noisy_authentic[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_clean_waveform_authentic = stft_to_waveform(real_clean_authentic[vis_idx], device=self.device).cpu().numpy().squeeze()

            self.logger.experiment.log({"fake_clean_waveform_authentic": [wandb.Audio(fake_clean_waveform_authentic, sample_rate=16000, caption="Generated Clean Audio - Authentic")]})
            self.logger.experiment.log({"mask_waveform_authentic": [wandb.Audio(mask_waveform_authentic, sample_rate=16000, caption="Learned Mask by Generator - Authentic")]})
            self.logger.experiment.log({"real_noisy_waveform_authentic": [wandb.Audio(real_noisy_waveform_authentic, sample_rate=16000, caption="Original Noisy Audio - Authentic")]})
            self.logger.experiment.log({"real_clean_waveform_authentic": [wandb.Audio(real_clean_waveform_authentic, sample_rate=16000, caption="Original Clean Audio - Authentic")]})

            plt = visualize_stft_spectrogram(real_clean_waveform_authentic, fake_clean_waveform_authentic, real_noisy_waveform_authentic)
            self.logger.experiment.log({"Spectrogram_Authentic": [wandb.Image(plt, caption="Spectrogram - Authentic")]})
            plt.close()

            plt = visualize_stft_spectrogram(mask_waveform_paired, np.zeros_like(mask_waveform_paired), np.zeros_like(mask_waveform_paired))
            self.logger.experiment.log({"Mask_Paired": [wandb.Image(plt, caption="Mask - Paired")]})
            plt.close()
 
            plt = visualize_stft_spectrogram(mask_waveform_authentic, np.zeros_like(mask_waveform_authentic), np.zeros_like(mask_waveform_authentic))
            self.logger.experiment.log({"Mask_Authentic": [wandb.Image(plt, caption="Mask - Authentic")]})
            plt.close()

if __name__ == "__main__":
    # Print Device
    print(torch.cuda.is_available())

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

    model = AutoencoderMix(discriminator=Discriminator(), generator=Generator(), visualize=False,
                        alpha_penalty=10,
                        alpha_fidelity=10,

                        n_critic=1,
                        use_bias=True,
                        
                        d_learning_rate=1e-4,
                        d_scheduler_step_size=1000,
                        d_scheduler_gamma=1,

                        g_learning_rate=1e-4,
                        g_scheduler_step_size=1000,
                        g_scheduler_gamma=1,

                        weight_clip = False,
                        weight_clip_value=0.5,

                        logging_freq=5,
                        batch_size=4,
                        log_all_scores=True,
                        L2_reg = False,
                        val_fraction = 1.)
    
    trainer = L.Trainer(max_epochs=5, accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=20, limit_val_batches=0,
                        logger=False,
                        fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)
