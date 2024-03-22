from pytorch_lightning.utilities.types import STEP_OUTPUT, TRAIN_DATALOADERS
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.utils.utils import stft_to_waveform, perfect_shuffle, visualize_stft_spectrogram
import pytorch_lightning as L
import torch
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True
import wandb
import csv
# from pesq import pesq


# define the Autoencoder class containing the training setup
class Autoencoder(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

        self.discriminator=Discriminator(use_bias=self.use_bias)
        self.generator=Generator(in_channels=2, out_channels=2)

        # save hyperparameters to Weights and Biases
        self.save_hyperparameters(kwargs)
        self.automatic_optimization = False
        self.csv_file = open('disc_weights.csv', 'a')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(["conv0", "conv1", "conv2", "conv3", "conv4", "conv5", "linear0", "linear1"])

    def forward(self, real_noisy):
        return self.generator(real_noisy)

    def _get_reconstruction_loss(self, real_noisy, fake_clean, D_fake, p=1):
        # Compute the Lp loss between the real clean and the fake clean
        G_fidelity_loss = torch.norm(fake_clean - real_noisy, p=p)
        # Normalize the loss by the number of elements in the tensor
        G_fidelity_loss = G_fidelity_loss / (real_noisy.size(0) * real_noisy.size(3))
        # compute adversarial loss
        G_adv_loss = - torch.mean(D_fake)
        # # Compute the total generator loss
        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss
        G_loss /= self.n_critic

        return G_loss, self.alpha_fidelity * G_fidelity_loss, G_adv_loss
    
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
        grad_norms = torch.sqrt(torch.sum(gradients ** 2, dim=1) + 1e-10) 
        gradient_penalty = torch.mean((grad_norms - 1.) ** 2)

        # Adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()

        # Total discriminator loss
        D_loss = self.alpha_penalty * gradient_penalty + D_adv_loss

        return D_loss, self.alpha_penalty * gradient_penalty, D_adv_loss
        
    def configure_optimizers(self):
        g_opt = torch.optim.Adam(self.generator.parameters(), lr=self.g_learning_rate)
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)

        return [g_opt, d_opt], []
    
    def bias_regularizer(self, module, lambda_=0.01):
        l2_reg = torch.tensor(0., device=self.device)
        for name, param in module.named_parameters():
            if 'bias' in name:
                l2_reg += lambda_ * torch.norm(param)
        return l2_reg
            
    def training_step(self, batch, batch_idx):
        g_opt, d_opt = self.optimizers()
        # g_sch, d_sch = self.lr_schedulers()

        d_opt.zero_grad()
        g_opt.zero_grad()

        real_clean = batch[0].squeeze(1)
        real_noisy = batch[1].squeeze(1)

        fake_clean, mask = self.generator(real_noisy)

        D_real = self.discriminator(real_clean)
        D_fake = self.discriminator(fake_clean)
        # detach fake_clean to avoid computing gradients for the generator when computing discriminator loss
        D_fake_no_grad = self.discriminator(fake_clean.detach())

        D_loss, D_gp_alpha, D_adv_loss = self._get_discriminator_loss(real_clean=real_clean, fake_clean=fake_clean, D_real=D_real, D_fake_no_grad=D_fake_no_grad)
        G_loss, G_fidelity_alpha, G_adv_loss = self._get_reconstruction_loss(real_noisy=real_noisy, fake_clean=fake_clean, D_fake=D_fake)

        # Backward pass
        self.manual_backward(D_loss, retain_graph=True)

        if batch_idx % self.n_critic == 0 and self.current_epoch >= 0 and batch_idx != 0:
            self.manual_backward(G_loss)

        d_opt.step()

        if batch_idx % self.n_critic == 0 and self.current_epoch >= 0 and batch_idx != 0:
            g_opt.step()

        # Weight clipping
        if self.weight_clip:
            for p in self.discriminator.parameters():
                p.data.clamp_(-self.weight_clip_value, self.weight_clip_value)
            
        # Update learning rate every epoch
        # if self.trainer.is_last_batch:
        #     g_sch.step()
        #     d_sch.step()

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

        fake_clean, mask = self(real_noisy)

        real_clean_waveforms = stft_to_waveform(real_clean, device=self.device).detach().cpu().squeeze()
        fake_clean_waveforms = stft_to_waveform(fake_clean, device=self.device).detach().cpu().squeeze()
        real_noisy_waveforms = stft_to_waveform(real_noisy, device=self.device).detach().cpu().squeeze()

        ## Scale Invariant Signal-to-Noise Ratio
        sisnr = ScaleInvariantSignalNoiseRatio().to(self.device)
        sisnr_score = sisnr(preds=fake_clean_waveforms, target=real_clean_waveforms)
        self.log('SI-SNR', sisnr_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        # SI-SNR for noisy = 8.753

        ## Extended Short Time Objective Intelligibility
        estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
        estoi_score = estoi(preds = fake_clean_waveforms, target = real_clean_waveforms)
        self.log('eSTOI', estoi_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        if self.log_all_scores:
            ## Mean Opinion Score (SQUIM)
            if self.current_epoch % 10 == 0 and batch_idx % 10 == 0:
                reference_waveforms = perfect_shuffle(real_clean_waveforms)
                subjective_model = SQUIM_SUBJECTIVE.get_model()
                mos_squim_score = torch.mean(subjective_model(fake_clean_waveforms, reference_waveforms)).item()
                self.log('MOS SQUIM', mos_squim_score, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            
            ## Predicted objective metrics: STOI, PESQ, and SI-SDR
            objective_model = SQUIM_OBJECTIVE.get_model()
            stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveforms)
            self.log('STOI Pred', stoi_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('PESQ Pred', pesq_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log('SI-SDR Pred', si_sdr_pred.mean(), on_step=False, on_epoch=True, prog_bar=True, logger=True)


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

                        n_critic=10,
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
                        log_all_scores=False)
    
    trainer = L.Trainer(max_epochs=5, accelerator='auto', num_sanity_val_steps=0,
                        log_every_n_steps=1, limit_train_batches=20, limit_val_batches=0,
                        logger=False,
                        fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)
