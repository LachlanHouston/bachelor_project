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

# Import data
from gan import data_loader

torch.manual_seed(42)
#wandb.init(False)

@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    # Print device
    print(torch.cuda.is_available())

    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.clean_processed_path)
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), cfg.data.noisy_processed_path)

    # Load the data loaders
    train_loader, val_loader, test_loader = data_loader(clean_path, noisy_path, cfg.data.split, cfg.hyperparameters.batch_size, cfg.hyperparameters.num_workers)
    print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))

    model = Autoencoder(discriminator=Discriminator(), 
                        generator=Generator(), 
                        alpha_penalty=cfg.hyperparameters.alpha_penalty,
                        alpha_fidelity=cfg.hyperparameters.alpha_fidelity,
                        n_critic=cfg.hyperparameters.n_critic,
                        logging_freq=cfg.wandb.logging_freq)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}-{val_SNR:.2f}",  # Checkpoint file name
        save_top_k=1,  # Save the top k models
        verbose=True,  # Print a message when a checkpoint is saved
        monitor="val_SNR",  # Metric to monitor for deciding the best model
        mode="max",  # Mode for the monitored quantity for model selection
    )

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=5,
        logger=L.loggers.WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,
        ),
        callbacks=[checkpoint_callback],
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
    print("Done!")