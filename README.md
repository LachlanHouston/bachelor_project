# Deep Learning for Speech Enhancement

Implementation of a Wasserstein GAN with gradient penalty to perform speech enhancement

## Environment setup

The conda environment can be setup by using make commands. The following command will create a new conda environment with the necessary dependencies:

`make create_environment` will create a new conda environment with the necessary dependencies (from the `requirements.txt` file). Remember to activate the environment before running the code.

## Data

The data used in this project is the VCTKD dataset, which can be downloaded from [here](https://datashare.ed.ac.uk/handle/10283/2791). The data should be placed in the `data` directory in the `clean_raw` and `noisy_raw` folders, while test data should be placed in `test_clean_raw` and `test_noisy_raw`. Other audio datasets should be placed in another folder in the `data` directory. The data should be in the form of `.wav` files.

## Training the model

The training script can be either run from an IDE or by using the `make train` command. The training script will train the model using the configuration specified in the `gan/config/config.yaml` file. The training script will save the model checkpoints in the `models` directory.

## Location of important code

All code can be found in the `gan` directory.

`gan/config/config.yaml` contains the configuration for the model, including hyperparameters and wandb settings.

`gan/train_model.py` contains the training script for the model using PyTorch Lightning. If using an IDE, this script can be run to train the model.

`gan/models/autoencoder.py` contains the training loop, including loss functions and training and validation steps.

`gan/models/discriminator.py` and `gan/models/generator.py` contain the implementation of the discriminator and generator.

## WANDB

To log the training process, we use WANDB. To use WANDB, you need to create an account and create a secrets.env file under the personal folder, if the personal folder does not exist, create it under the primary folder. The secrets.env file should contain the following: WAND_API_KEY=YOUR_API_KEY. The WANDB API key can be found in your account settings. This is gitignored to prevent the key from being uploaded to the repository.

## Repository file structure

The directory structure of the project looks like this:

```txt
┌── data                 <- Where data should be located (VCTKD can be downloaded from https://datashare.ed.ac.uk/handle/10283/2791)
│   ├── clean_raw        <- Clean VCTKD training data
│   ├── noisy_raw        <- Noisy VCTKD training data
│   └── AudioSet         <- Where AudioSet data from the "speech" class should be located
│       ├── train_raw    <- AudioSet training data
│       └── test_raw     <- AudioSet test data
│
├── gan  <- All scripts used in this project
│   │
│   ├── config             
│   │   └── config.yaml  <- Configuration file to specify hyperparameters
│   │
│   ├── data             
│   │   └── data_loader.py     <- Script to load data to the GAN
│   │
│   ├── models           <- Model implementations, GAN training implementation
│   │   ├── autoencoder.py     <- GAN training loop containing training step, validation step, and loss functions
│   │   ├── generator.py       <- Generator model
│   │   ├── discriminator.py   <- Discriminator model
│   │   └── DPRNN.py           <- Dual-path block in the discriminator model
│   │
│   ├── utils            <- Utility scripts
│   │   ├── plot_learning_curve.py      <- Plot learning curves
│   │   └── utils.py                    <- Helper functions
│   │
│   ├── visualizations   <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py <- Visualize different aspects of the GAN and results
│   │
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
├── models               <- Most important model checkpoints
│
├── reports              <- Generated figures and scores with relevant subfolders
│
├── Makefile             <- Makefile with convenience commands like `make train`
│
├── personal             <- Folder for personal files like secrets.env (gitignored)
│
├── README.md            <- README detailing the structure of this repository
│
└── requirements.txt     <- The requirements file for reproducing the analysis environment
```


