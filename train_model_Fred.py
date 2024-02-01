import torch
from torch.utils.data import DataLoader
from scipy.stats import wasserstein_distance
import hydra
import os
import wandb
from omegaconf import OmegaConf
import torchaudio
from tqdm import tqdm
torch.manual_seed(42)
from data.data_loader import AudioDataset, collate_fn
# Import models
from gan import Generator, Discriminator, get_discriminator_loss

wandb.init(mode="disabled")

@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    # Load the waveform
    clean_path = os.path.join(hydra.utils.get_original_cwd(), cfg.clean_processed_path)
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), cfg.noisy_processed_path)
    dataset = AudioDataset(clean_path, noisy_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    D_optimizer = None
    G_optimizer = None

    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    epoch = 0
    discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])
    generator = Generator()

    # Model training
    for idx, (real_noisy, real_clean) in enumerate(tqdm(loader, leave=True)):
        # real_noisy = real_noisy.to(cfg.device)

        # Get outputs of discriminator and generator
        fake_clean = generator(real_noisy[0])
        D_real = discriminator(real_clean)
        D_fake = discriminator(fake_clean)

        # Train the discriminator
        D_loss = get_discriminator_loss(D_real, D_fake, discriminator, alpha=cfg.alpha_gp)
        # Update discriminator weights
        D_optimizer.zero_grad()
        D_loss.backward()
        D_optimizer.step()

        # Train the generator
        G_adv_loss = - torch.mean(D_fake)
        # wasserstein distance between noisy and generated:
        # G_fidelity_loss = wasserstein_distance(real_noisy.squeeze(0), fake_clean.squeeze(0))
        fake_clean_cat = torch.cat((fake_clean, fake_clean), dim=1)
        real_noisy_cat = torch.cat((real_noisy, real_noisy), dim=1)
        G_fidelity_loss = torch.norm(fake_clean_cat - real_noisy_cat, p=2)**2

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