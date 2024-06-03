import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import hydra
import os
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
# Import data
from gan import AudioDataModule, DummyDataModule, MixDataModule, SpeakerDataModule, FinetuneDataModule


# main function using Hydra to organize configuration
@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    # Print GPU information
    print('CUDA available:', torch.cuda.is_available())

    seeds = [50]
    #fractions = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    fractions = [1, 0.9, 0.8, 0.7, 0.6]

    for seed in seeds:
        print(f"Using seed {seed}")
        L.seed_everything(seed, workers=True)
        for fraction in fractions:
            import wandb
            print(f"Training with {fraction} of data")

            # configure wandb
            wandb_api_key = os.environ.get("WANDB_API_KEY")
            wandb.login(key=wandb_api_key)

            # define paths
            VCTK_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_raw/')
            VCTK_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_raw/')
            VCTK_test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_raw/')
            VCTK_test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_raw/')

            data_module = AudioDataModule(clean_path = VCTK_clean_path,
                                            noisy_path = VCTK_noisy_path,
                                            test_clean_path = VCTK_test_clean_path,
                                            test_noisy_path = VCTK_test_noisy_path,
                                            batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = fraction)      

            print("Using a single dataset")
            from gan import Autoencoder
                
                
            model = Autoencoder(alpha_penalty =         cfg.hyperparameters.alpha_penalty,
                                alpha_fidelity =        cfg.hyperparameters.alpha_fidelity,

                                n_critic =              cfg.hyperparameters.n_critic,
                             
                                d_learning_rate =       cfg.hyperparameters.d_learning_rate,
                                g_learning_rate =       cfg.hyperparameters.g_learning_rate,
                                log_all_scores =        cfg.wandb.log_all_scores,
                                batch_size =            cfg.hyperparameters.batch_size,
                                sisnr_loss =            cfg.hyperparameters.sisnr_loss,
                                val_fraction =          cfg.hyperparameters.val_fraction,
                                dataset =               cfg.hyperparameters.dataset,
                                ckpt_path =             cfg.system.ckpt_path,
                                )
            
            # define saving of checkpoints
            checkpoint_callback = ModelCheckpoint(
                save_top_k = 30,
                dirpath="models/",  # path where checkpoints will be saved
                every_n_epochs=1,  # how often to save a model checkpoint
                monitor='SI-SNR',  # quantity to monitor
                mode='max',  # mode to monitor
                filename="epoch-{epoch}-SISNR-{SI-SNR}",  # name of the checkpoint file
                save_on_train_epoch_end=True,
                )
            
            # define Weights and Biases logger
            wandb_logger = WandbLogger(
                project=cfg.wandb.project,
                name=cfg.wandb.name,
                entity=cfg.wandb.entity, 
            )

            # define the trainer 
            trainer = Trainer(
                accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                devices=cfg.hyperparameters.num_gpus if cfg.hyperparameters.num_gpus >= 1 and torch.cuda.is_available() else 'auto',
                strategy='ddp_find_unused_parameters_true' if cfg.hyperparameters.num_gpus > 1 and torch.cuda.is_available() else 'auto',
                max_epochs=cfg.hyperparameters.max_epochs,
                check_val_every_n_epoch=25,
                logger=wandb_logger,
                callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
                profiler=cfg.system.profiler if cfg.system.profiler else None,
                deterministic=True,
                limit_val_batches=cfg.hyperparameters.val_fraction,
                limit_train_batches=cfg.hyperparameters.train_fraction if cfg.hyperparameters.dataset == 'dummy' else 1.0,
                num_sanity_val_steps=2,
            )
            
            # train the model. Continue training from the last checkpoint if specified in config
            if cfg.system.continue_training and not cfg.hyperparameters.load_generator_only:
                print("Continuing training from checkpoint")
                trainer.fit(model, data_module, ckpt_path=os.path.join(hydra.utils.get_original_cwd(), cfg.system.ckpt_path))
            
            else:
                print("Starting new training")
                trainer.fit(model, data_module)


if __name__ == "__main__":
    main()
    print("Done!")
