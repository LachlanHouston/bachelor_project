import torch
torch.manual_seed(42)
import hydra
import os
import wandb
from omegaconf import OmegaConf
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler
import warnings
warnings.filterwarnings("ignore")
# Import models
from gan import Generator, Discriminator
from gan import Autoencoder
# Import data
from gan import data_loader

@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_processed/')
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_processed/')
    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_processed/')
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_processed/')

    # Load the data loaders
    train_loader, val_loader = data_loader( clean_path, noisy_path, 
                                            test_clean_path, test_noisy_path,
                                            cfg.hyperparameters.batch_size, cfg.hyperparameters.num_workers if torch.cuda.is_available() else 1)
    print('Train:', len(train_loader), 'Validation:', len(val_loader))

    model = Autoencoder(discriminator=Discriminator(), 
                        generator=Generator(), 
                        alpha_penalty=cfg.hyperparameters.alpha_penalty,
                        alpha_fidelity=cfg.hyperparameters.alpha_fidelity,
                        n_critic=cfg.hyperparameters.n_critic,
                        logging_freq=cfg.wandb.logging_freq,
                        d_learning_rate=cfg.hyperparameters.d_learning_rate,
                        d_scheduler_gamma=cfg.hyperparameters.d_scheduler_gamma,
                        g_learning_rate=cfg.hyperparameters.g_learning_rate,
                        g_scheduler_gamma=cfg.hyperparameters.g_scheduler_gamma,
                        weight_clip = cfg.hyperparameters.weight_clip,
                        weight_clip_value=cfg.hyperparameters.weight_clip_value,
                        visualize=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}-{val_SNR:.2f}",  # Checkpoint file name
        save_top_k=1,  # Save the top k models
        verbose=True,  # Print a message when a checkpoint is saved
        monitor="val_SNR",  # Metric to monitor for deciding the best model
        mode="max",  # Mode for the monitored quantity for model selection
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
    )

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        #precision="bf16-mixed",
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        profiler=profiler
    )

    # log gradients and model topology
    wandb_logger.watch(model)

    trainer.fit(model, train_loader, val_loader)
    # trainer.test(model, test_loader)


if __name__ == "__main__":
    main()
    print("Done!")