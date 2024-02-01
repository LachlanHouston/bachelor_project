import torch
import hydra
import os
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

# Import models
from gan import Generator, Discriminator
from gan import Autoencoder

torch.manual_seed(42)
#wandb.init(False)

@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    device = "cuda" if torch.cuda.is_available() and cfg.device == "cuda" else "cpu"
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.clean_processed_path)
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.noisy_processed_path)

    model = Autoencoder(discriminator=Discriminator(), 
                        generator=Generator(), 
                        batch_size=cfg.hyperparameters.batch_size, 
                        num_workers=cfg.hyperparameters.num_workers,
                        alpha_penalty=cfg.hyperparameters.alpha_penalty,
                        alpha_fidelity=cfg.hyperparameters.alpha_fidelity,
                        clean_path=clean_path,
                        noisy_path=noisy_path)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}-{val_acc:.2f}",  # Checkpoint file name
        save_top_k=1,  # Save the top k models
        verbose=True,  # Print a message when a checkpoint is saved
        every_n_epochs = 1 # Save checkpoint every n epochs
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=L.loggers.WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
        ),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model)


if __name__ == "__main__":
    main()
    print("Done!")