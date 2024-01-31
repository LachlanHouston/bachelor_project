import torch
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
import hydra
import os
import wandb
from omegaconf import OmegaConf
from tqdm import tqdm
torch.manual_seed(42)
from data.data_loader import AudioDataset, collate_fn
# Import models
from gan import Generator, Discriminator



@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg, D_optimizer, G_optimizer):

    dataset = AudioDataset(cfg.clean_processed_path, cfg.noisy_processed_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    D_optimizer = None
    G_optimizer = None

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    epoch = 0
    discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])

    # Model training
    for idx, (real_noisy, real_clean) in enumerate(tqdm(loader, leave=True)):
        real_noisy = real_noisy.to(cfg.device)

        # Get outputs of discriminator and generator
        fake_clean = Generator(real_noisy)
        D_real = Discriminator(real_clean)
        D_fake = Discriminator(fake_clean)

        # Train the discriminator
        D_loss = get_discriminator_loss(D_real, D_fake, discriminator, alpha=cfg.alpha_gp)
        # Update discriminator weights
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train the generator
        G_adv_loss = - torch.mean(D_fake)
        # wasserstein distance between noisy and generated:
        G_fidelity_loss = wasserstein_distance(real_noisy.squeeze(0), fake_clean.squeeze(0))

        G_loss = cfg.alpha_fidelity * G_fidelity_loss + G_adv_loss

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