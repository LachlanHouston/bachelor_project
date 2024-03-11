from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.utils.utils import stft_to_waveform, perfect_shuffle, visualize_stft_spectrogram
import pytorch_lightning as L
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_SUBJECTIVE
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
import wandb
# from pesq import pesq


# define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)
        # save hyperparameters to Weights and Biases
        self.save_hyperparameters(kwargs)
        self.automatic_optimization = False

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
        # gradient penalty
        alpha = torch.rand(self.batch_size, 1, 1, 1, device=self.device) # B x 1 x 1 x 1
        alpha = alpha.expand_as(real_clean).to(self.device) # B x C x H x W
        differences = fake_clean - real_clean # B x C x H x W
        interpolates = real_clean + (alpha * differences) # B x C x H x W
        interpolates.requires_grad_(True)
        D_interpolates = self.discriminator(interpolates) # B x 1 (the output of the discriminator is a scalar value for each input sample)
        ones = torch.ones(D_interpolates.size(), device=self.device) # B x 1
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W
        gradients = gradients.view(self.batch_size, -1) # B x (C*H*W)
        grad_norms = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-10) 
        gradient_penalty = torch.mean((grad_norms - 1.) ** 2)

        # adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()
        # total discriminator loss
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
        if batch_idx % self.n_critic == 0 and self.current_epoch > 1:
            g_opt.zero_grad()

        real_clean = batch[0].squeeze(1)
        real_noisy = batch[1].squeeze(1)
            
        # real_clean = torch.normal(0, 1, (4, 2, 257, 321))
        # real_noisy = torch.normal(0, 1, (4, 2, 257, 321))

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

        if batch_idx % self.n_critic == 0 and self.current_epoch > 1:
            g_opt.step()

        # Weight clipping
        if self.weight_clip:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)
            
        # Update learning rate every epoch
        if self.trainer.is_last_batch:
            g_sch.step()
            d_sch.step()

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
            
        # real_clean = torch.normal(0, 1, (4, 2, 257, 321))
        # real_noisy = torch.normal(0, 1, (4, 2, 257, 321))        

        fake_clean, mask = self.generator(real_noisy)

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).detach().cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).detach().cpu().squeeze()
        real_noisy_waveforms = stft_to_waveform(real_noisy, device=self.device).detach().cpu().squeeze()


        ## Scale Invariant Signal-to-Noise Ratio
        sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
        sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
        self.log('SI-SNR', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        sisnr_noisy = ScaleInvariantSignalNoiseRatio().to(self.device)
        sisnr_noisy_score = sisnr_noisy(preds=real_noisy_waveforms, target=real_clean_waveforms)
        self.log('SI-SNR Noisy', sisnr_noisy_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        ## Extended Short Time Objective Intelligibility
        estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
        estoi_score = estoi(preds = fake_clean_waveforms, target = real_clean_waveforms)
        self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        ## Mean Opinion Score (SQUIM)
        if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
            reference_waveforms = perfect_shuffle(real_clean_waveforms)
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
            self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ## Perceptual Evaluation of Speech Quality
        # pesq_scores = [pesq(fs=16000, ref=real_clean_waveforms[i].numpy(), deg=fake_clean_waveforms[i].numpy(), mode='wb') for i in range(self.batch_size)]
        # pesq_score = torch.tensor(pesq_scores).mean()
        # self.log('PESQ', pesq_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)


        # visualize the spectrogram and waveforms every first batch of every self.logging_freq epochs
        if batch_idx == 0 and self.current_epoch % self.logging_freq == 0:
            vis_idx = torch.randint(0, self.batch_size, (1,)).item()
            # log spectrograms
            plt = visualize_stft_spectrogram(real_clean[vis_idx], fake_clean[vis_idx], real_noisy[vis_idx])
            self.logger.experiment.log({"Spectrogram": [wandb.Image(plt, caption="Spectrogram")]})
            plt.close()
            # log waveforms
            fake_clean_waveform = stft_to_waveform(fake_clean[vis_idx], device=self.device).detach().cpu().numpy().squeeze()
            mask_waveform = stft_to_waveform(mask[vis_idx], device=self.device).detach().cpu().numpy().squeeze()
            real_noisy_waveform = stft_to_waveform(real_noisy[vis_idx], device=self.device).detach().cpu().numpy().squeeze()
            real_clean_waveform = stft_to_waveform(real_clean[vis_idx], device=self.device).detach().cpu().numpy().squeeze()
            self.logger.experiment.log({"fake_clean_waveform": [wandb.Audio(fake_clean_waveform, sample_rate=16000, caption="Generated Clean Audio")]})
            self.logger.experiment.log({"mask_waveform": [wandb.Audio(mask_waveform, sample_rate=16000, caption="Learned Mask by Generator")]})
            self.logger.experiment.log({"real_noisy_waveform": [wandb.Audio(real_noisy_waveform, sample_rate=16000, caption="Original Noisy Audio")]})
            self.logger.experiment.log({"real_clean_waveform": [wandb.Audio(real_clean_waveform, sample_rate=16000, caption="Original Clean Audio")]})

            plt = visualize_stft_spectrogram(mask[vis_idx], torch.zeros_like(mask[vis_idx]), torch.zeros_like(mask[vis_idx]))
            self.logger.experiment.log({"Mask": [wandb.Image(plt, caption="Mask")]})
            plt.close()


if __name__ == "__main__":
    # Print Device
    print(torch.cuda.is_available())

    # Dummy train_loader
    train_loader = torch.utils.data.DataLoader(
        [torch.randn(2, 257, 321), torch.randn(2, 257, 321)],
        batch_size=10,
        shuffle=True
    )

    val_loader = torch.utils.data.DataLoader(
        [torch.randn(2, 257, 321), torch.randn(2, 257, 321)],
        batch_size=10,
        shuffle=True
    )

    model = Autoencoder(discriminator=Discriminator(), generator=Generator(), visualize=False)
    trainer = L.Trainer(max_epochs=5, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=5, limit_val_batches=1,
                        logger=False)
    trainer.fit(model, train_loader, val_loader)
