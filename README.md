# Deep Learning for Speech Enhancement

Implementation of a Wasserstein GAN with gradient penalty to perform speech enhancement

## Location of important code

All code can be found in the `gan` directory.

`gan/models/autoencoder.py` contains the training loop, including loss functions and training and validation steps.

`gan/models/discriminator.py` and `gan/models/generator.py` contain the implementation of the discriminator and generator.

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
│   │   └── visualize.py
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
├── README.md            <- README detailing the structure of this repository
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
└── requirements_dev.txt <- The requirements file for reproducing the analysis environment
```


