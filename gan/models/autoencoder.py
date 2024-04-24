from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.utils.utils import stft_to_waveform, perfect_shuffle, visualize_stft_spectrogram
import pytorch_lightning as L
import torch
from torch.optim.swa_utils import AveragedModel, SWALR
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
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.discriminator=Discriminator(use_bias=self.use_bias).to(self.device)
        self.generator=Generator(in_channels=2, out_channels=2).to(self.device)
        self.custom_global_step = 0
        self.save_hyperparameters(kwargs) # save hyperparameters to Weights and Biases
        self.automatic_optimization = False
        self.example_input_array = torch.randn(self.batch_size, 2, 257, 321)

    def forward(self, real_noisy):
        if len(real_noisy[0].shape) == 5:
            batch = real_noisy
            real_clean = batch[0].squeeze(1).to(self.device)
            real_noisy = batch[1].squeeze(1).to(self.device)
            return real_clean, self.generator(real_noisy)
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, real_clean, p=1):
        if self.supervised_fidelity:
            # Compute the Lp loss between the real clean and the fake clean
            G_fidelity_loss = torch.norm(fake_clean - real_clean, p=p)
            # Normalize the loss by the number of elements in the tensor
            G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))
        else:
            # Compute the Lp loss between the real noisy and the fake clean
            G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
            # Normalize the loss by the number of elements in the tensor
            G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))
        
        # compute adversarial loss
        G_adv_loss = - torch.mean(D_fake)
        # Compute the total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss

        if self.sisnr_loss:
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
        alpha = torch.rand(self.batch_size, 1, 1, 1, device=self.device) # B x 1 x 1 x 1
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
        gradients = gradients.view(self.batch_size, -1) # B x (C*H*W)
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
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)#, betas = (0., 0.9))
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)#, betas = (0., 0.9))
        return [g_opt, d_opt], []
    
    def training_step(self, batch, batch_idx):

        g_opt, d_opt = self.optimizers()

        train_G = (self.custom_global_step + 1) % self.n_critic == 0

        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)

        if (self.swa_start_epoch_g is not False) and self.current_epoch == self.swa_start_epoch_g and batch_idx == 0:
            self.swa_generator = AveragedModel(self.generator)

        if train_G:
            self.toggle_optimizer(g_opt)
            # Generate fake clean
            fake_clean, mask = self.generator(real_noisy)
            D_fake = self.discriminator(fake_clean)
            G_loss, G_fidelity_alpha, G_adv_loss, sisnr_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, D_fake=D_fake, real_clean=real_clean)
            # Compute generator gradients
            self.manual_backward(G_loss)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)
            
            # Update SWA weights
            if (self.swa_start_epoch_g is not False) and (self.current_epoch >= self.swa_start_epoch_g) and (batch_idx == 0):
                self.swa_generator.update_parameters(self.generator)

        self.toggle_optimizer(d_opt)
        fake_clean_no_grad = self.generator(real_noisy)[0].detach()
        D_fake_no_grad = self.discriminator(fake_clean_no_grad)
        D_real = self.discriminator(real_clean)
        D_loss, D_gp_alpha, D_adv_loss, L2_penalty_bias = self._get_discriminator_loss(real_clean=real_clean, fake_clean=fake_clean_no_grad, D_real=D_real, D_fake_no_grad=D_fake_no_grad)
        # Compute discriminator gradients
        self.manual_backward(D_loss)
        # Update discriminator weights
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        D_fake = D_fake_no_grad
        # log discriminator losses
        self.log('D_Loss', D_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Real', D_real.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake', D_fake.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Penalty', D_gp_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if self.L2_reg:
            self.log('D_bias_penalty', L2_penalty_bias, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        if train_G:
            # Log generator losses
            self.log('G_Loss', G_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            self.log('G_Adversarial', G_adv_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True) # opposite sign as D_fake
            self.log('G_Fidelity', G_fidelity_alpha, on_step=True, on_epoch=False, prog_bar=True, logger=True)
            if self.sisnr_loss:
                self.log('G_SI-SNR_Loss', sisnr_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

        if self.custom_global_step % 10 == 0 and self.dataset == "VCTK":        
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SNR training', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.log_all_scores and self.custom_global_step % 50 == 0:
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            ## Predicted objective metric: SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('SI-SDR pred Training', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI pred Training', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ pred Training', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        self.custom_global_step += 1


    def on_train_epoch_end(self):
        # Check if SWA is being used and if it's past the starting epoch
        if (self.swa_start_epoch_g is not False) and self.current_epoch >= self.swa_start_epoch_g:
            # Update Batch Normalization statistics for the swa_generator
            torch.optim.swa_utils.update_bn(self.trainer.train_dataloader, self.swa_generator)
            # Now the swa_generator is ready to be used for validation


    def validation_step(self, batch, batch_idx):
        # Remove tuples and convert to tensors
        real_clean = batch[0]
        real_noisy = batch[1]     

        # Check if SWA is being used and if it's past the starting epoch
        if (self.swa_start_epoch_g is not False) and self.current_epoch >= self.swa_start_epoch_g:
            fake_clean, mask = self.swa_generator(real_noisy)
        else:
            fake_clean, mask = self.generator(real_noisy)

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()

        if self.dataset == "VCTK":
            ## Scale Invariant Signal-to-Noise Ratio
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SNR', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            # SI-SNR for noisy = 8.753

            ## Extended Short Time Objective Intelligibility
            estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
            estoi_score = estoi(preds = fake_clean_waveforms, target = real_clean_waveforms)
            self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Mean Opinion Score (SQUIM)
        if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
            reference_waveforms = perfect_shuffle(real_clean_waveforms)
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
            self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if (self.log_all_scores or self.dataset != "VCTK") and batch_idx % 50 == 0:
            ## Predicted objective metrics: STOI, PESQ, and SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('SI-SDR Pred', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI Pred', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)


        # visualize the spectrogram and waveforms every first batch of every self.logging_freq epochs
        if batch_idx == 0:
            self.vis_batch_idx = torch.randint(0, (int(824*self.val_fraction)) // self.batch_size, (1,)).item()
        if batch_idx == self.vis_batch_idx and self.current_epoch % self.logging_freq == 0:
            vis_idx = torch.randint(0, self.batch_size, (1,)).item()
            # log waveforms
            fake_clean_waveform = stft_to_waveform(fake_clean[vis_idx], device=self.device).cpu().numpy().squeeze()
            mask_waveform = stft_to_waveform(mask[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_noisy_waveform = stft_to_waveform(real_noisy[vis_idx], device=self.device).cpu().numpy().squeeze()
            real_clean_waveform = stft_to_waveform(real_clean[vis_idx], device=self.device).cpu().numpy().squeeze()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(fake_clean_waveform, sample_rate=16000, caption="Generated Clean Audio")]})
            self.logger.experiment.log({"mask_waveform": [wandb.Audio(mask_waveform, sample_rate=16000, caption="Learned Mask by Generator")]})
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

    model = Autoencoder(discriminator=Discriminator(), generator=Generator(), visualize=False,
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
