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

torch.set_float32_matmul_precision('medium')
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
torch.backends.cuda.matmul.allow_tf32 = True

@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_stft/')
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_stft/')
    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_stft/')
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_stft/')

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
                        
                        d_learning_rate=cfg.hyperparameters.d_learning_rate,
                        d_scheduler_step_size=cfg.hyperparameters.d_scheduler_step_size,
                        d_scheduler_gamma=cfg.hyperparameters.d_scheduler_gamma,

                        g_learning_rate=cfg.hyperparameters.g_learning_rate,
                        g_scheduler_step_size=cfg.hyperparameters.g_scheduler_step_size,
                        g_scheduler_gamma=cfg.hyperparameters.g_scheduler_gamma,

                        weight_clip = cfg.hyperparameters.weight_clip,
                        weight_clip_value=cfg.hyperparameters.weight_clip_value,

                        visualize=True,
                        logging_freq=cfg.wandb.logging_freq)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}",  # The name of the checkpoint files
    )

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity,
    )

    profiler = AdvancedProfiler(dirpath=".", filename="perf_logs")

    trainer = Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        limit_train_batches=cfg.hyperparameters.train_fraction,
        limit_val_batches= cfg.hyperparameters.val_fraction,
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback],
        profiler="simple"
    )

    # log gradients and model topology
    wandb_logger.watch(model)

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
    print("Done!")