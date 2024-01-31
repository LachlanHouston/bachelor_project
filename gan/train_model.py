import torch
from scipy.stats import wasserstein_distance
import hydra
import os
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
torch.manual_seed(42)
# Import models
from models.generator import Generator
from models.discriminator import Discriminator

loader = None
D_optimizer = None
G_optimizer = None


@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg, loader, D_optimizer, G_optimizer):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    epoch = 0
    
    # Model training
    for idx, (real_noisy, real_clean) in enumerate(tqdm(loader)):
        real_noisy = real_noisy.to(cfg.device)
        fake_clean = Generator(real_noisy)

        D_real = Discriminator(real_clean)
        D_fake = Discriminator(fake_clean)
        # Train the discriminator
        D_real_mean = torch.mean(D_real)
        D_fake_mean = torch.mean(D_fake)
        D_adv_loss = D_fake_mean - D_real_mean
        D_loss = D_adv_loss + cfg.alpha_gp * gradient_penalty(D_real, real_clean, D_fake, fake_clean)
        # Update discriminator weights
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train the generator
        G_adv_loss = None
        # wasserstein distance between noisy and generated:
        G_fidelity_loss = wasserstein_distance(input.squeeze(0), output.squeeze(0))

        G_loss = cfg.alpha_fidelity_loss * G_fidelity_loss + G_adv_loss

        # Update generator weights
        G_optimizer.zero_grad()
        G_loss.backward()
        G_optimizer.step()

        if idx % 100 == 0:
            print(f"epoch: {epoch} \nDiscriminator loss: {D_loss.item()} \nGenerator loss: {G_loss.item()}")
            wandb.log({"Discriminator loss": D_loss.item()})
            wandb.log({"Generator loss": G_loss.item()})
        if idx % 1000 == 0 and cfg.save_model:
            model = None
            torch.save(model.state_dict(), f"checkpoints/{idx}.pt")
            wandb.save(f"checkpoints/{idx}.pt")

if __name__ == "__main__":
    main()