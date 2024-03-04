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
from pytorch_lightning.tuner.tuning import Tuner
from lightning.pytorch.profilers import AdvancedProfiler
import warnings
warnings.filterwarnings("ignore")
# Import models
from gan import Generator, Discriminator
from gan import Autoencoder
# Import data
from gan import VCTKDataModule

# torch.set_float32_matmul_precision('medium')
# torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
# torch.backends.cuda.matmul.allow_tf32 = True


@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    L.seed_everything(100)
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # Define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_stft/')
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_stft/')
    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_stft/')
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_stft/')

    # Load the data loaders
    VCTK = VCTKDataModule(clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.hyperparameters.num_workers)

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
                        logging_freq=cfg.wandb.logging_freq,
                        batch_size=cfg.hyperparameters.batch_size,
                        )
    
    checkpoint_callback = ModelCheckpoint(
        dirpath="models/",  # Path where checkpoints will be saved
        filename="{epoch}",  # The name of the checkpoint files
        every_n_epochs=5,  # Save a checkpoint every epoch

    )

    if cfg.wandb.use_wandb:
        wandb_logger = WandbLogger(
            project=cfg.wandb.project,
            name=cfg.wandb.name,
            entity=cfg.wandb.entity,  
        )
        # log gradients and model topology
        wandb_logger.watch(model)
    else:
        wandb_logger = None

    if cfg.system.num_gpus >= 1:
        trainer = Trainer(
            accelerator="cuda" if torch.cuda.is_available() else "cpu",
            devices=cfg.system.num_gpus,
            strategy="ddp_find_unused_parameters_true" if cfg.system.num_gpus > 1 else "auto",
            limit_train_batches=cfg.hyperparameters.train_fraction,
            limit_val_batches= cfg.hyperparameters.val_fraction,
            max_epochs=cfg.hyperparameters.max_epochs,
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
        )

    else:
        trainer = Trainer(
            accelerator="cpu",
            limit_train_batches=cfg.hyperparameters.train_fraction,
            limit_val_batches= cfg.hyperparameters.val_fraction,
            max_epochs=cfg.hyperparameters.max_epochs,
            check_val_every_n_epoch=1,
            logger=wandb_logger,
            callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
        )

    # tuner = Tuner(trainer)
    # tuner.scale_batch_size(model, VCTK)

    if cfg.system.continue_training:
        print("Continuing training from checkpoint")
        trainer.fit(model, VCTK, ckpt_path=os.path.join(hydra.utils.get_original_cwd(), cfg.system.ckpt_path))

    else:
        print("Starting new training")
        trainer.fit(model, VCTK)


if __name__ == "__main__":
    main()
    print("Done!")