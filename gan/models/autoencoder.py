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
import os
# from pesq import pesq

# define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        
        self.discriminator=Discriminator().to(self.device)
        self.generator=Generator(in_channels=2, out_channels=2).to(self.device)
        if self.load_generator_only:
            print('Loading only the generator from checkpoint')
            cwd = '/'.join(os.getcwd().split('/')[:-3])
            print('cwd:', cwd)
            print('path:', os.path.join(cwd, self.ckpt_path))
            checkpoint = torch.load(os.path.join(cwd, self.ckpt_path), map_location=self.device)
            generator_keys = {k.replace('model.', ''): v 
                for k, v in checkpoint['state_dict'].items() 
                if k.startswith('model.')}
            self.generator.load_state_dict(generator_keys, strict=False)
            print("Generator loaded from checkpoint")

        self.custom_global_step = 0
        self.save_hyperparameters(kwargs) # save hyperparameters to Weights and Biases
        self.automatic_optimization = False
        # self.example_input_array = torch.randn(self.batch_size, 2, 257, 321)

    def forward(self, real_noisy):
        if len(real_noisy[0].shape) == 5:
            batch = real_noisy
            real_clean = batch[0].squeeze(1).to(self.device)
            real_noisy = batch[1].squeeze(1).to(self.device)
            return real_clean, self.generator(real_noisy)
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, real_clean, p=1):
        # Compute the Lp loss between the real noisy and the fake clean
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # Normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))
        # compute adversarial loss
        G_adv_loss = - torch.mean(D_fake)
        # Compute the total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss

        if self.sisnr_loss or self.dataset == 'Finetune':
            if self.sisnr_loss is False:
                self.sisnr_loss = 10
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()
            if self.dataset == 'AudioSet':
                objective_model = SQUIM_OBJECTIVE.get_model()
                _, _, si_sdr_pred = objective_model(fake_clean_waveforms)
                for i in range(len(si_sdr_pred)):
                    if torch.isnan(torch.tensor([si_sdr_pred[i]])).any():
                        print(f"si_sdr_pred contains NaN values.")
                    
                sisnr_loss = - si_sdr_pred.mean()
            else:
                sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
                if self.sisnr_loss_half_batch:
                    sisnr_loss = - sisnr(preds=fake_clean_waveforms[:self.batch_size//2], target=real_clean_waveforms[:self.batch_size//2])
                else:
                    sisnr_loss = - sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            # Multiply the sisnr loss by a scaling factor
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

        return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss
    

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

        if self.swa_start_epoch_g is not False:
            self.swa_scheduler = SWALR(g_opt, swa_lr=self.swa_lr)

        if not self.linear_lr_scheduling:
            g_lr_scheduler = torch.optim.lr_scheduler.StepLR(g_opt, step_size=self.g_scheduler_step_size, gamma=self.g_scheduler_gamma)
            d_lr_scheduler = torch.optim.lr_scheduler.StepLR(d_opt, step_size=self.d_scheduler_step_size, gamma=self.d_scheduler_gamma)
            return ({"optimizer": g_opt, "lr_scheduler": g_lr_scheduler},
                    {"optimizer": d_opt, "lr_scheduler": d_lr_scheduler})
        
        start_lr, end_lr, total_iters = self.linear_lr_scheduling
        start_factor_g, start_factor_d = start_lr / self.g_learning_rate,   start_lr / self.d_learning_rate
        end_factor_g, end_factor_d =     end_lr / self.g_learning_rate,     end_lr / self.d_learning_rate
        g_lr_scheduler = torch.optim.lr_scheduler.LinearLR(g_opt, start_factor=start_factor_g, end_factor=end_factor_g, total_iters=total_iters, verbose=True)
        d_lr_scheduler = torch.optim.lr_scheduler.LinearLR(d_opt, start_factor=start_factor_d, end_factor=end_factor_d, total_iters=total_iters, verbose=True)
        torch.optim.lr_scheduler.ExponentialLR()
        return ({"optimizer": g_opt,"lr_scheduler": g_lr_scheduler},
                {"optimizer": d_opt, "lr_scheduler": d_lr_scheduler})


    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        if self.linear_lr_scheduling:
            self.g_lr_scheduler, self.d_lr_scheduler = self.lr_schedulers()

        train_G = (self.custom_global_step + 1) % self.n_critic == 0

        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)
        # noisy_name = batch[3]

        if train_G:
            self.toggle_optimizer(g_opt)
            # Generate fake clean
            fake_clean, mask = self.generator(real_noisy)
            if torch.isnan(fake_clean).any():
                print(f"Output of generator contains NaN values. Noisy input: {noisy_name}")
            D_fake = self.discriminator(fake_clean)
            G_loss, G_fidelity_alpha, G_adv_loss, sisnr_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, D_fake=D_fake, real_clean=real_clean)
            # Compute generator gradients
            self.manual_backward(G_loss)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        fake_clean_no_grad = self.generator(real_noisy)[0].detach()
        if torch.isnan(fake_clean_no_grad).any():
            print(f"Output of generator contains NaN values. Noisy input: {noisy_name}")
        D_fake_no_grad = self.discriminator(fake_clean_no_grad)
        D_real = self.discriminator(real_clean)
        D_loss, D_gp_alpha, D_adv_loss = self._get_discriminator_loss(real_clean=real_clean, fake_clean=fake_clean_no_grad, D_real=D_real, D_fake_no_grad=D_fake_no_grad)
        # Compute discriminator gradients
        self.manual_backward(D_loss)
        # Update discriminator weights
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        D_fake = D_fake_no_grad
        # log discriminator losses
        self.log('D_Loss', D_loss,        on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('D_Real', D_real.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('D_Fake', D_fake.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('D_Penalty', D_gp_alpha, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # Log generator losses
        if train_G:
            self.log('G_Loss', G_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('G_Adversarial', G_adv_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True) # opposite sign as D_fake
            self.log('G_Fidelity', G_fidelity_alpha, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            if self.sisnr_loss:
                self.log('G_SI-SNR_Loss', sisnr_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.custom_global_step % 10 == 0 and self.dataset == "VCTK":        
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SNR training', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            if self.sisnr_loss_half_batch:
                sisnr_score_supervised = sisnr(preds=fake_clean_waveforms[:self.batch_size//2], target=real_clean_waveforms[:self.batch_size//2])
                self.log('SI-SNR supervised training', sisnr_score_supervised, on_step=False, on_epoch=True, prog_bar=True, logger=True)
                sisnr_score_unsupervised = sisnr(preds=fake_clean_waveforms[self.batch_size//2:], target=real_clean_waveforms[self.batch_size//2:])
                self.log('SI-SNR unsupervised training', sisnr_score_unsupervised, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.log_all_scores and self.custom_global_step % 50 == 0:
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            ## Predicted objective metric: SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('SI-SDR pred Training', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('STOI pred Training', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ pred Training', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if (self.swa_start_epoch_g is not False) and (not hasattr(self, 'swa_generator')):
            self.swa_generator = AveragedModel(self.generator)
        self.custom_global_step += 1


    def on_train_epoch_end(self):

        # Check if SWA is being used and if it's past the starting epoch
        if (self.swa_start_epoch_g is not False) and self.current_epoch >= self.swa_start_epoch_g:
            # Update SWA weights
            self.swa_generator.update_parameters(self.generator)

            self.swa_scheduler.step()
            print("Updating SWA BN")
            # Update Batch Normalization statistics for the swa_generator
            torch.optim.swa_utils.update_bn(self.trainer.val_dataloaders, self.swa_generator, device=self.device)
            # Now the swa_generator is ready to be used for validation
            print("SWA BN done")
            # Save SWA generator checkpoint every 5 epochs
            if (self.current_epoch+1) % 10 == 0:
                # Create the directory if it doesn't exist
                if not os.path.exists('models'):
                    os.makedirs('models')
                # Save the SWA generator checkpoint
                torch.save(self.swa_generator.state_dict(), 'models/swa_generator_epoch_{}.ckpt'.format(self.current_epoch))

        # Otherwise, step the learning rate schedulers normally
        else:
            old_lr = self.optimizers()[0].param_groups[0]['lr']
            self.lr_schedulers()[0].step()
            self.lr_schedulers()[1].step()
            new_lr = self.optimizers()[0].param_groups[0]['lr']
            if old_lr != new_lr:
                print('G learning rate:', self.optimizers()[0].param_groups[0]['lr'], 
                    '\n D learning rate:', self.optimizers()[1].param_groups[0]['lr'])

        # Log the norms of the generator and discriminator parameters
        if self.current_epoch % 1 == 0:
            for name, param in self.generator.named_parameters():
                self.log(f'Generator_{name}_norm', param.norm(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            for name, param in self.discriminator.named_parameters():
                self.log(f'Discriminator_{name}_norm', param.norm(), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            # Also log the overall norm of the generator and discriminator parameters
            self.log('Generator_mean_norm', torch.norm(torch.cat([param.view(-1) for param in self.generator.parameters()])), on_step=False, on_epoch=True, prog_bar=False, logger=True)
            self.log('Discriminator_mean_norm', torch.norm(torch.cat([param.view(-1) for param in self.discriminator.parameters()])), on_step=False, on_epoch=True, prog_bar=False, logger=True)


    def validation_step(self, batch, batch_idx):
        # Remove tuples and convert to tensors
        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)
        # real_clean_name = batch[2]
        # real_noisy_name = batch[3]

        nan_count_clean = torch.isnan(real_clean).sum().item()
        if nan_count_clean > 0:
            raise ValueError(f"Detected {nan_count_clean} NaN values in real_clean data")
        nan_count_noisy = torch.isnan(real_noisy).sum().item()
        if nan_count_noisy > 0:
            raise ValueError(f"Detected {nan_count_noisy} NaN values in real_noisy data")

        # Check if SWA is being used and if it's past the starting epoch
        if (self.swa_start_epoch_g is not False) and self.current_epoch >= self.swa_start_epoch_g:
            fake_clean, mask = self.swa_generator(real_noisy)
        else:
            fake_clean, mask = self.generator(real_noisy)

        nan_count_fake = torch.isnan(fake_clean).sum().item()
        if nan_count_fake > 0:
            raise ValueError(f"Detected {nan_count_fake} NaN values in fake_clean data")

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()

        if self.dataset == "VCTK" or "Speaker":
            ## Scale Invariant Signal-to-Noise Ratio
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log('SI-SNR', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            ## Extended Short Time Objective Intelligibility
            estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
            estoi_score = estoi(preds = fake_clean_waveforms, target = real_clean_waveforms)
            self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.current_epoch % 10 == 0 and batch_idx % 5 == 0:
            ## Mean Opinion Score (SQUIM)
            reference_waveforms = perfect_shuffle(real_clean_waveforms)
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
            self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        
        if (self.log_all_scores or (self.dataset not in ["VCTK", "Speaker"])) and batch_idx % 10 == 0:
            ## Predicted objective metrics: STOI, PESQ, and SI-SDR
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
            try:
                plt = visualize_stft_spectrogram(real_clean_waveform, fake_clean_waveform, real_noisy_waveform)
            except:
                print("Error in visualizing spectrograms")
                print('real_clean:', real_clean_name[vis_idx])
                print('real_noisy:', real_noisy_name[vis_idx])
                self.logger.experiment.log({"Problematic_fake_clean_waveform": [wandb.Audio(fake_clean_waveform, sample_rate=16000, caption=f"{real_clean_name[vis_idx]}")]})
                self.logger.experiment.log({"Problematic_real_noisy_waveform": [wandb.Audio(real_noisy_waveform, sample_rate=16000, caption=f"{real_noisy_name[vis_idx]}" )]})
                self.logger.experiment.log({"Problematic_real_clean_waveform": [wandb.Audio(real_clean_waveform, sample_rate=16000, caption=f"{real_clean_name[vis_idx]}")]})


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

    model = Autoencoder(discriminator=Discriminator(), 
                        generator=Generator(),
                        alpha_penalty=10,
                        alpha_fidelity=10,

                        n_critic=1,
                        
                        d_learning_rate=1e-4,
                        d_scheduler_step_size=1000,
                        d_scheduler_gamma=1,

                        g_learning_rate=1e-4,
                        g_scheduler_step_size=1000,
                        g_scheduler_gamma=1,

                        batch_size=4,
                        log_all_scores=True,
                        val_fraction = 1.)
    
    trainer = L.Trainer(max_epochs=5, accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=20, limit_val_batches=0,
                        logger=False,
                        fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)
