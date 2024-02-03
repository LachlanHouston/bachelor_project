from typing import Any
from pytorch_lightning.utilities.types import STEP_OUTPUT
from gan.models.generator import Generator
from gan.models.discriminator import Discriminator
from gan.data.data_loader import data_loader
import pytorch_lightning as L
import torch
from torchmetrics.audio import SignalNoiseRatio

class Autoencoder(L.LightningModule):
    def __init__(self, 
                    discriminator: Discriminator = None,
                    generator: Generator = None,
                    alpha_penalty=10,
                    alpha_fidelity=10
                 ):
        super().__init__()
        self.generator = Generator()
        self.discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])
        self.alpha_penalty = alpha_penalty
        self.alpha_fidelity = alpha_fidelity

    def forward(self, real_clean, real_noisy):
        d_real = self.discriminator(real_clean)
        fake_clean = self.generator(real_noisy)
        d_fake = self.discriminator(fake_clean)
        return d_real, d_fake, fake_clean
    
    def _get_reconstruction_loss(self, d_fake, fake_clean, real_noisy):
        G_adv_loss = - torch.mean(d_fake)
        fake_clean_cat = torch.cat((fake_clean, fake_clean), dim=1)
        real_noisy_cat = torch.cat((real_noisy, real_noisy), dim=1)
        G_fidelity_loss = torch.norm(fake_clean_cat - real_noisy_cat, p=2)**2

        G_loss = self.alpha_fidelity * G_fidelity_loss + G_adv_loss
        return G_loss
    
    def _get_discriminator_loss(self, d_real, d_fake, real_input, fake_input):
        alpha = torch.rand(real_input.size(0), 1, 1, 1, device=self.device)

        difference = (fake_input - real_input)
        interpolates = (real_input + (alpha * difference))
        interpolates = interpolates
        
        out = self.discriminator(interpolates)
        grad_outputs = torch.ones(out.size(), device=self.device)

        gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=grad_outputs, create_graph=True, retain_graph=True, only_inputs=True)[0]
        slopes = torch.sqrt(torch.sum(torch.square(gradients), axis=[1, 2, 3]))
        gradient_penalty = torch.mean((slopes - 1.) ** 2)

        D_adv_loss = d_fake.mean() - d_real.mean()
        D_loss = D_adv_loss + self.alpha_penalty * gradient_penalty

        return D_loss
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)

        return optimizer

    def training_step(self, batch, batch_idx):
        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        d_real, d_fake, fake_clean = self.forward(real_clean, real_noisy)

        # Train the discriminator
        D_loss = self._get_discriminator_loss(d_real, d_fake, real_clean, fake_clean)
        
        # Train the generator
        G_loss = self._get_reconstruction_loss(d_fake, fake_clean, real_noisy)

        # Compute total loss
        loss = D_loss + G_loss
        
        self.log('D_loss', D_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('G_loss', G_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Combined loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        # Compute batch SNR
        real_clean = batch[0]
        real_noisy = batch[1]

        # Remove tuples and convert to tensors
        real_clean = torch.stack(real_clean, dim=1).squeeze(0)
        real_noisy = torch.stack(real_noisy, dim=1).squeeze(0)

        fake_clean = self.generator(real_noisy)

        # Signal to Noise Ratio
        snr = SignalNoiseRatio()
        snr_val = snr(fake_clean, real_clean)

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
        snr = SignalNoiseRatio()
        snr_val = snr(fake_clean, real_clean)

        self.log('test_SNR', snr_val)

if __name__ == "__main__":
    # Print Device
    print(torch.cuda.is_available())
    train_loader, val_loader, test_loader = data_loader('data/clean_processed/', 'data/noisy_processed/', batch_size=4, num_workers=4)
    print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    model = Autoencoder(discriminator=Discriminator(), generator=Generator())
    trainer = L.Trainer(max_epochs=1, accelerator='auto', num_sanity_val_steps=2,
                        log_every_n_steps=1, limit_train_batches=10, limit_val_batches=10, limit_test_batches=10)
    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)

    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard

