import os
import torch
import hydra
import wandb
import pytorch_lightning as L
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
import warnings
warnings.filterwarnings("ignore")
from gan import Autoencoder, AudioDataModule, DummyDataModule, SpeakerDataModule, FinetuneDataModule, pl_Discriminator


# main function using Hydra to organize configuration
@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    # Print GPU information
    print(torch.cuda.is_available())

    L.seed_everything(100, workers=True)
    # configure wandb
    wandb_api_key = os.environ.get("WANDB_API_KEY")
    wandb.login(key=wandb_api_key)

    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity, 
    )

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
        print("Using dummy data")
        data_module = DummyDataModule(batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers)
    elif cfg.hyperparameters.dataset == "VCTK":
        print("Using VCTK data")
        data_module = AudioDataModule(VCTK_clean_path, VCTK_noisy_path, VCTK_test_clean_path, VCTK_test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers, fraction=cfg.hyperparameters.train_fraction, authentic=False)
    elif cfg.hyperparameters.dataset == "AudioSet":
        print("Using AudioSet data")
        # use AudioSet as noisy data and VCTK as clean data
        data_module = AudioDataModule(VCTK_clean_path, AudioSet_noisy_path, VCTK_test_clean_path, AudioSet_test_noisy_path, batch_size=cfg.hyperparameters.batch_size, num_workers=cfg.system.num_workers, fraction=cfg.hyperparameters.train_fraction, authentic=True)
    elif cfg.hyperparameters.dataset == "Speaker":
        print("Using data with a custom number of speakers")
        data_module = SpeakerDataModule(clean_path = VCTK_clean_path,
                                        noisy_path = VCTK_noisy_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = cfg.hyperparameters.train_fraction, num_speakers=cfg.hyperparameters.num_speakers)      
    elif cfg.hyperparameters.dataset == "Finetune":
        print("Using data for finetuning")
        data_module = FinetuneDataModule(clean_path = VCTK_clean_finetune_path,
                                        noisy_path = VCTK_noisy_finetune_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, num_speakers=cfg.hyperparameters.num_speakers, fraction = cfg.hyperparameters.train_fraction)
    elif cfg.hyperparameters.dataset == "Unsuper50p":
        print("Using data with 50 percent of the speakers for unsupervised training")
        data_module = AudioDataModule(clean_path = VCTK_unsuper50p_clean_path,
                                        noisy_path = VCTK_unsuper50p_noisy_path,
                                        test_clean_path = VCTK_test_clean_path,
                                        test_noisy_path = VCTK_test_noisy_path,
                                        batch_size = cfg.hyperparameters.batch_size, num_workers = cfg.system.num_workers, fraction = cfg.hyperparameters.train_fraction)


    model = pl_Discriminator(batch_size=cfg.hyperparameters.batch_size, d_learning_rate=cfg.hyperparameters.d_learning_rate, 
                             alpha_penalty=cfg.hyperparameters.alpha_penalty, use_bias=cfg.hyperparameters.use_bias)
    
    # define saving of checkpoints
    checkpoint_callback = ModelCheckpoint(
        save_top_k = -1,  # save all checkpoints
        dirpath="models/",  # path where checkpoints will be saved
        filename="{epoch}",  # the name of the checkpoint files
        every_n_epochs=25,  # how often to save a model checkpoint
    )

    # define Weights and Biases logger
    wandb_logger = WandbLogger(
        project=cfg.wandb.project,
        name=cfg.wandb.name,
        entity=cfg.wandb.entity, 
    )

    # log gradients and model topology
    wandb_logger.watch(model, log='all', log_freq=1)

    # define the trainer 
    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=cfg.hyperparameters.num_gpus if cfg.hyperparameters.num_gpus >= 1 and torch.cuda.is_available() else 'auto',
        strategy='ddp_find_unused_parameters_true' if cfg.hyperparameters.num_gpus > 1 and torch.cuda.is_available() else 'auto',
        max_epochs=cfg.hyperparameters.max_epochs,
        check_val_every_n_epoch=1,
        logger=wandb_logger,
        callbacks=[checkpoint_callback] if cfg.system.checkpointing else None,
        profiler=cfg.system.profiler if cfg.system.profiler else None,
        deterministic=True,
        limit_val_batches=cfg.hyperparameters.val_fraction,
    )
    # train the model. Continue training from the last checkpoint if specified in config
    if cfg.system.continue_training:
        print("Continuing training from checkpoint")
        trainer.fit(model, data_module, ckpt_path=os.path.join(hydra.utils.get_original_cwd(), cfg.system.ckpt_path))
    else:
        print("Starting new training")
        trainer.fit(model, data_module)

if __name__ == "__main__":
    main()
    print("Done!")