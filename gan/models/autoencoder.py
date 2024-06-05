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


# define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        # define the discriminator and generator
        self.discriminator=Discriminator().to(self.device)
        self.generator=Generator(in_channels=2, out_channels=2).to(self.device)
        self.custom_global_step = 0
        self.save_hyperparameters(kwargs) # save hyperparameters to Weights and Biases
        self.automatic_optimization = False

    def forward(self, real_noisy):
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, real_clean, p=1):
        # compute the Lp loss between the real noisy and the fake clean
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / (real_noisy.size(1) * real_noisy.size(2) * real_noisy.size(3))
        # adversarial loss
        G_adv_loss = - torch.mean(D_fake)
        # total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss
        # extra supervised SI-SNR loss term if specified in the config
        if self.sisnr_loss or self.dataset == 'Finetune':
            if self.sisnr_loss is False:
                self.sisnr_loss = 10 # set to 10 in case dataset is "Finetune" and it is not already set
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()
            # use SI-SDR instead of SI-SNR if dataset is AudioSet
            if self.dataset == 'AudioSet':
                objective_model = SQUIM_OBJECTIVE.get_model()
                _, _, si_sdr_pred = objective_model(fake_clean_waveforms)
                # define the loss as the negative mean of the SI-SDR                    
                sisnr_loss = - si_sdr_pred.mean()
            else:
                sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
                # define the loss as the negative mean of the SI-SNR
                sisnr_loss = - sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            # multiply the SI-SNR loss by a scaling factor specified in the config
            sisnr_loss *= self.sisnr_loss
            # add the SI-SNR loss to the total generator loss
            G_loss += sisnr_loss
            return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss, sisnr_loss

<<<<<<< Updated upstream
        return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss, None
    
    def _get_discriminator_loss(self, real_clean, fake_clean, D_real, D_fake_no_grad):
        # create interpolated samples
        alpha = torch.rand(self.batch_size, 1, 1, 1, device=self.device) # B x 1 x 1 x 1
        # alpha = alpha.expand(real_clean.size()) # B x C x H x W
        differences = fake_clean - real_clean # B x C x H x W
        interpolates = real_clean + (alpha * differences) # B x C x H x W
        interpolates.requires_grad_(True)
=======
        # Compute SI-SDR loss
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device_).cpu().squeeze()
        objective_model = SQUIM_OBJECTIVE.get_model()
        _, _, si_sdr_pred = objective_model(fake_clean_waveforms)
        # Define the loss as the negative mean of the SI-SDR
        sisnr_loss = -si_sdr_pred.mean()
        sisnr_loss *= self.sisnr_loss
>>>>>>> Stashed changes

        # calculate the output of the discriminator for the interpolated samples and compute the gradients
        D_interpolates = self.discriminator(interpolates) # B x 1 (the output of the discriminator is a scalar value for each input sample)
        ones = torch.ones(D_interpolates.size(), device=self.device) # B x 1
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W
        
        # calculate the gradient penalty
        gradients = gradients.view(self.batch_size, -1) # B x (C*H*W)
        grad_norms = gradients.norm(2, dim=1) # B
        gradient_penalty = ((grad_norms - 1) ** 2).mean()
        # adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()
        # total discriminator loss
        D_loss = self.alpha_penalty * gradient_penalty + D_adv_loss
        return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss
    

    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)
        return g_opt, d_opt

    # Helper function to log values to Weights and Biases
    def log_value(self, name, value):
        self.log(name, value, on_step=False, on_epoch=True, prog_bar=True, logger=True)


    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        # unpack batched data
        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)

        train_G = (self.custom_global_step + 1) % self.n_critic == 0
        if train_G:
            self.toggle_optimizer(g_opt)
            # generate fake clean
            fake_clean, mask = self.generator(real_noisy)
            D_fake = self.discriminator(fake_clean)
            G_loss, G_fidelity_alpha, G_adv_loss, sisnr_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, D_fake=D_fake, real_clean=real_clean)
            # compute generator gradients
            self.manual_backward(G_loss)
            g_opt.step()
            g_opt.zero_grad()
            self.untoggle_optimizer(g_opt)

        self.toggle_optimizer(d_opt)
        fake_clean_no_grad = self.generator(real_noisy)[0].detach()
        D_fake_no_grad = self.discriminator(fake_clean_no_grad)
        D_real = self.discriminator(real_clean)
        D_loss, D_gp_alpha, D_adv_loss = self._get_discriminator_loss(real_clean=real_clean, fake_clean=fake_clean_no_grad, D_real=D_real, D_fake_no_grad=D_fake_no_grad)
        # compute discriminator gradients
        self.manual_backward(D_loss)
        # update discriminator weights
        d_opt.step()
        d_opt.zero_grad()
        self.untoggle_optimizer(d_opt)

        D_fake = D_fake_no_grad

        # log discriminator losses
        self.log_value('D_Loss', D_loss)
        self.log_value('D_Real', D_real.mean())
        self.log_value('D_Fake', D_fake.mean())
        self.log_value('D_Penalty', D_gp_alpha)

        # log generator losses
        if train_G:
            self.log_value('G_Loss', G_loss)
            self.log_value('G_Adversarial', G_adv_loss)
            self.log_value('G_Fidelity', G_fidelity_alpha)
            if self.sisnr_loss:
                self.log_value('G_SI-SNR_Loss', sisnr_loss)

        if self.custom_global_step % 10 == 0 and self.dataset == "VCTK":
            real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            sisnr_score = ScaleInvariantSignalNoiseRatio().to(self.device)(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log_value('SI-SNR training', sisnr_score)

        if self.log_all_scores and self.custom_global_step % 10 == 0:
            fake_clean_waveforms = stft_to_waveform(fake_clean_no_grad, device=self.device).cpu().squeeze()
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log_value('SI-SDR pred Training', si_sdr_pred.mean())
            self.log_value('STOI pred Training', stoi_pred.mean())
            self.log_value('PESQ pred Training', pesq_pred.mean())
        
        self.custom_global_step += 1


    def on_train_epoch_end(self):
        # log the norms of the generator and discriminator parameters
        if self.current_epoch % 1 == 0:
            for name, param in self.generator.named_parameters():
                self.log_value(f'Weight Norms/Gen_{name}_norm', param.norm())
            for name, param in self.discriminator.named_parameters():
                self.log_value(f'Weight Norms/Disc_{name}_norm', param.norm())
            # also log the overall norm of the generator and discriminator parameters
            gen_mean_norm = torch.norm(torch.cat([param.view(-1) for param in self.generator.parameters()]))
            disc_mean_norm = torch.norm(torch.cat([param.view(-1) for param in self.discriminator.parameters()]))
            self.log_value('Weight Norms/Gen_mean_norm', gen_mean_norm)
            self.log_value('Weight Norms/Disc_mean_norm', disc_mean_norm)


    def validation_step(self, batch, batch_idx):
        # remove tuples and convert to tensors
        real_clean = batch[0].to(self.device)
        real_noisy = batch[1].to(self.device)
        # generate fake clean
        fake_clean, mask = self.generator(real_noisy)
        # convert to waveforms
        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).cpu().squeeze()

        if self.dataset != 'AudioSet':
            # Scale Invariant Signal-to-Noise Ratio
            sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
            sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log_value('SI-SNR', sisnr_score)

            # Extended Short Time Objective Intelligibility
            estoi = ShortTimeObjectiveIntelligibility(16000, extended=True)
            estoi_score = estoi(preds=fake_clean_waveforms, target=real_clean_waveforms)
            self.log_value('eSTOI', estoi_score)

        if self.current_epoch % 10 == 0 and batch_idx % 5 == 0:
            # Mean Opinion Score (SQUIM)
            reference_waveforms = perfect_shuffle(real_clean_waveforms)
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
            self.log_value('MOS SQUIM', mos_squim_score)

        if (self.log_all_scores or self.dataset == "AudioSet") and batch_idx % 10 == 0:
            # Predicted objective metrics: STOI, PESQ, and SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            preds = {'SI-SDR Pred': si_sdr_pred.mean(), 'STOI Pred': stoi_pred.mean(), 'PESQ Pred': pesq_pred.mean()}
            for name, value in preds.items():
                self.log_value(name, value)

        # visualize the waveforms and spectrograms
        if batch_idx == 0:
            self.vis_batch_idx = torch.randint(0, (int(824 * self.val_fraction)) // self.batch_size, (1,)).item() if self.dataset != 'dummy' else 0
        if batch_idx == self.vis_batch_idx:
            vis_idx = torch.randint(0, self.batch_size, (1,)).item() if self.dataset != 'dummy' else 0
            # log waveforms
            waveforms = {"fake_clean_waveform": stft_to_waveform(fake_clean[vis_idx], device=self.device).cpu().numpy().squeeze(),
                         "mask_waveform": stft_to_waveform(mask[vis_idx], device=self.device).cpu().numpy().squeeze(),
                         "real_noisy_waveform": stft_to_waveform(real_noisy[vis_idx], device=self.device).cpu().numpy().squeeze(),
                         "real_clean_waveform": stft_to_waveform(real_clean[vis_idx], device=self.device).cpu().numpy().squeeze()}
            for caption, waveform in waveforms.items():
                self.logger.experiment.log({caption: [wandb.Audio(waveform, sample_rate=16000, caption=f"{caption.replace('_', ' ').title()}")]})

            # log spectrograms
            plt = visualize_stft_spectrogram(waveforms["real_clean_waveform"], waveforms["fake_clean_waveform"], waveforms["real_noisy_waveform"])
            self.logger.experiment.log({"Spectrogram": [wandb.Image(plt, caption="Spectrogram")]})
            plt.close()
            plt = visualize_stft_spectrogram(waveforms["mask_waveform"], np.zeros_like(waveforms["mask_waveform"]), np.zeros_like(waveforms["mask_waveform"]))
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
