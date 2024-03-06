import torch
torch.manual_seed(42)
import hydra
import os
import wandb
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
# Import models
from gan import Generator, Discriminator
from gan import Autoencoder
# Import data
from gan import VCTKDataModule

# main function using Hydra to organize configuration
@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    L.seed_everything(100)
    # configure wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    # define paths
    clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_stft/')
    noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_stft/')
    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_stft/')
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_stft/')

    # load the data loaders
    VCTK = VCTKDataModule(clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.hyperparameters.num_workers)

    # define the autoencoder class containing the training setup
    model = Autoencoder(discriminator=Discriminator(), 
                        generator=Generator(in_channels=2, out_channels=2),

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
    
    # define saving of checkpoints
    checkpoint_callback = ModelCheckpoint(
        save_top_k = -1,  # save all checkpoints
        dirpath="models/",  # path where checkpoints will be saved
        filename="{epoch}",  # the name of the checkpoint files
        every_n_epochs=5,  # how often to save a model checkpoint
    )

    # define Weights and Biases logger
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

    # define the trainer 
    trainer = Trainer(
        accelerator='cuda' if torch.cuda.is_available() else 'cpu',
        devices=cfg.system.num_gpus if cfg.system.num_gpus >= 1 and torch.cuda.is_available() else 'auto',
        strategy='ddp_find_unused_parameters_true' if cfg.system.num_gpus > 1 and torch.cuda.is_available() else 'auto',
        limit_train_batches=cfg.hyperparameters.train_fraction,
        limit_val_batches= cfg.hyperparameters.val_fraction,
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
        profiler=cfg.system.profiler if cfg.system.profiler else None,
    )

    # train the model. Continue training from the last checkpoint if specified in config
    if cfg.system.continue_training:
        print("Continuing training from checkpoint")
        trainer.fit(model, VCTK, ckpt_path=os.path.join(hydra.utils.get_original_cwd(), cfg.system.ckpt_path))
    else:
        print("Starting new training")
        trainer.fit(model, VCTK)

    # save profiling results
    if cfg.system.profiler:
        with open("profiling.txt", "w") as file:
            file.write(trainer.profiler.summary())

if __name__ == "__main__":
    main()
    print("Done!")