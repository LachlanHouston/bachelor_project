import torch.nn as nn
import torch
import pytorch_lightning as L

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), padding=(0, 0), use_bias=True):
        super().__init__()
        norm_f = nn.utils.spectral_norm
        self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias))
        self.activation = nn.LeakyReLU(0.1)
        
        nn.init.xavier_uniform_(self.conv.weight)
        # nn.init.kaiming_uniform_(self.conv.weight, a=0.1, mode='fan_in', nonlinearity='leaky_relu')
        if use_bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128], use_bias=True):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        norm_f = nn.utils.spectral_norm

        assert len(self.input_sizes) == len(self.output_sizes), "Input and output sizes must be the same length"

        for i in range(len(self.input_sizes)):
            self.conv_layers.append(Conv2DBlock(self.input_sizes[i], self.output_sizes[i], kernel_size=(5, 5), stride=(2, 2), use_bias=use_bias))

        self.activation = nn.LeakyReLU(0.1)    

        self.fc_layers1  = norm_f(nn.Linear(256, 64, bias=use_bias))
        self.fc_layers2 = norm_f(nn.Linear(64, 1, bias=use_bias))

    def forward(self, x) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
            
        x = x.flatten(1, -1)
        x = self.fc_layers1(x)
        x = self.activation(x)
        x = self.fc_layers2(x)

        return x
    
# PyTorch Lightning Module for discriminator only training
class pl_Discriminator(L.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            print(key, value)
            setattr(self, key, value)

        self.discriminator=Discriminator(use_bias=self.use_bias).to(self.device)

        # save hyperparameters to Weights and Biases
        self.save_hyperparameters(kwargs)

    def forward(self, batch): # Batch is a tuple of (real_clean, real_noisy)
        return [self.discriminator(batch[0]), self.discriminator(batch[1])]
    
    def _get_discriminator_loss(self, real_clean, fake_clean, D_real, D_fake_no_grad): 
        # Create interpolated samples
        alpha = torch.rand(self.batch_size, 1, 1, 1, device=self.device) # B x 1 x 1 x 1
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
        d_opt = torch.optim.Adam(self.discriminator.parameters(), lr=self.d_learning_rate)
        return d_opt
    
    def training_step(self, batch, batch_idx, *args):
        real_clean = batch[0].squeeze(1).to(self.device)
        real_noisy = batch[1].squeeze(1).to(self.device)

        # Train the discriminator
        D_clean = self.discriminator(real_clean)
        D_noisy = self.discriminator(real_noisy)

        self.log('D_Real', D_clean.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake', D_noisy.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Compute the discriminator loss
        D_loss, penalty, adv_loss = self._get_discriminator_loss(real_clean, real_noisy, D_clean, D_noisy)

        # Log the discriminator loss
        self.log('D_Loss', D_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Penalty', penalty, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('Adv_Loss', adv_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return D_loss

    def validation_step(self, batch, batch_idx, *args):
        torch.set_grad_enabled(True)
        real_clean = batch[0].squeeze(1).to(self.device)
        real_noisy = batch[1].squeeze(1).to(self.device)

        D_clean = self.discriminator(real_clean)
        D_noisy = self.discriminator(real_noisy)

        self.log('D_Real_val', D_clean.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('D_Fake_val', D_noisy.mean(), on_step=True, on_epoch=False, prog_bar=True, logger=True)

        # Compute the discriminator loss
        D_loss, penalty, adv_loss = self._get_discriminator_loss(real_clean, real_noisy, D_clean, D_noisy)

        # Log the discriminator loss
        self.log('D_Loss_val', D_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('Penalty_val', penalty, on_step=True, on_epoch=False, prog_bar=True, logger=True)
        self.log('Adv_Loss_val', adv_loss, on_step=True, on_epoch=False, prog_bar=True, logger=True)

if __name__ == '__main__':
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

    model = pl_Discriminator(batch_size=4, d_learning_rate=1e-4, alpha_penalty=10, use_bias=True)
    
    trainer = L.Trainer(max_epochs=5, accelerator='cuda' if torch.cuda.is_available() else 'cpu', num_sanity_val_steps=1,
                        log_every_n_steps=1, limit_train_batches=1, limit_val_batches=1,
                        logger=False,
                        fast_dev_run=False)
    trainer.fit(model, train_loader, val_loader)

        
        
