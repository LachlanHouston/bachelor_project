import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import torch
import hydra
import wandb
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
    print('MPS available:', torch.backends.mps.is_available())
    L.seed_everything(100, workers=True)

    # define paths
    VCTK_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_raw/')
    VCTK_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_raw/')
    VCTK_test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_raw/')
    VCTK_test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_raw/')
    VCTK_unsuper50p_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_raw_speakers/unsuper50p/')
    VCTK_unsuper50p_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_raw_speakers/unsuper50p/')
    VCTK_clean_finetune_path = os.path.join(hydra.utils.get_original_cwd(), 'data/clean_raw_speakers/super50p/')
    VCTK_noisy_finetune_path = os.path.join(hydra.utils.get_original_cwd(), 'data/noisy_raw_speakers/super50p/')

    AudioSet_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/AudioSet/train_raw/')
    AudioSet_test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/AudioSet/test_raw/')

    # load the data loaders
    if cfg.hyperparameters.dataset == "dummy":
        data_module = DummyDataModule(batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers)
    if cfg.hyperparameters.dataset == "VCTK":
        data_module = AudioDataModule(VCTK_clean_path, VCTK_noisy_path, VCTK_test_clean_path, VCTK_test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers, fraction=cfg.hyperparameters.train_fraction, authentic=False)
    if cfg.hyperparameters.dataset == "AudioSet":
        # use AudioSet as noisy data and VCTK as clean data
        data_module = AudioDataModule(VCTK_clean_path, AudioSet_noisy_path, VCTK_test_clean_path, AudioSet_test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers, fraction=cfg.hyperparameters.train_fraction, authentic=True)
    if cfg.hyperparameters.dataset == "Speaker":
        data_module = SpeakerDataModule(clean_path = VCTK_clean_path,
                                        noisy_path = VCTK_noisy_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = cfg.hyperparameters.train_fraction, num_speakers=cfg.hyperparameters.num_speakers)      
    if cfg.hyperparameters.dataset == "Finetune":
        data_module = FinetuneDataModule(clean_path = VCTK_clean_finetune_path,
                                        noisy_path = VCTK_noisy_finetune_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, num_speakers=cfg.hyperparameters.num_speakers, fraction = cfg.hyperparameters.train_fraction)
    if cfg.hyperparameters.dataset == "Unsuper50p":
        data_module = AudioDataModule(clean_path = VCTK_unsuper50p_clean_path,
                                        noisy_path = VCTK_unsuper50p_noisy_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = cfg.hyperparameters.train_fraction)

    if cfg.hyperparameters.dataset == "Mix":
        data_module = MixDataModule(clean_path = VCTK_clean_path,
                                    noisy_path_authentic = AudioSet_noisy_path,
                                    noisy_path_paired = VCTK_noisy_path,
                                    test_clean_path = VCTK_test_clean_path,
                                    test_noisy_path_authentic = AudioSet_test_noisy_path,
                                    test_noisy_path_paired = VCTK_test_noisy_path,
                                    batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = cfg.hyperparameters.train_fraction)
    
        # define the autoencoder class containing the training setup
        print("Using a mixture of authentic and paired data")
        from gan import AutoencoderMix as Autoencoder
    else:
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
        save_top_k = -1,  # save all checkpoints
        dirpath="models/",  # path where checkpoints will be saved
        filename="{epoch}",  # the name of the checkpoint files
        every_n_epochs=5,  # how often to save a model checkpoint
    )

    # configure wandb
    if cfg.wandb.use_wandb:
        wandb_api_key = os.environ.get("WANDB_API_KEY")
        wandb.login(key=wandb_api_key)
        # define Weights and Biases logger
        wandb_logger = WandbLogger(project=cfg.wandb.project, name=cfg.wandb.name, entity=cfg.wandb.entity)
        # log gradients and model topology
        wandb_logger.watch(model, log='gradients', log_freq=1)

    # define the trainer 
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.hyperparameters.num_gpus if cfg.hyperparameters.num_gpus >= 1 and torch.cuda.is_available() else 'auto',
        strategy='ddp_find_unused_parameters_true' if cfg.hyperparameters.num_gpus > 1 and torch.cuda.is_available() else 'auto',
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger if cfg.wandb.use_wandb else None,
        callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
        profiler=cfg.system.profiler if cfg.system.profiler else None,
        deterministic=True,
        limit_val_batches=cfg.hyperparameters.val_fraction,
        limit_train_batches=cfg.hyperparameters.train_fraction if cfg.hyperparameters.dataset == 'dummy' else 1.0,
        num_sanity_val_steps=2,
    )
    
    # train the model. Continue training from the last checkpoint if specified in config
    if cfg.system.continue_training:
        print("Continuing training from checkpoint")
        trainer.fit(model, data_module, ckpt_path=os.path.join(hydra.utils.get_original_cwd(), cfg.system.ckpt_path))
    
    else:
        print("Starting new training")
        trainer.fit(model, data_module)

    # save profiling results
    if cfg.system.profiler:
        with open("profiling.txt", "w") as file:
            file.write(trainer.profiler.summary())


if __name__ == "__main__":
    main()
    print("Done!")