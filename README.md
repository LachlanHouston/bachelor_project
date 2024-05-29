# Deep Learning for Speech Enhancement

Implementation of a Wasserstein GAN with gradient penalty to perform speech enhancement

## Project structure

The directory structure of the project looks like this:

```txt


├── data                 <- Where data should be located (VCTKD can be downloaded from https://datashare.ed.ac.uk/handle/10283/2791)
│   ├── clean_raw        <- Clean VCTKD training data
│   ├── noisy_raw        <- Noisy VCTKD training data
│   └── AudioSet         <- Where AudioSet data should be located
│       ├── train_raw    <- AudioSet training data
│       └── test_raw     <- AudioSet test data
│
├── gan  <- Source code for use in this project.
│   │
│   ├── data             <- Scripts to download or generate data
│   │   ├── __init__.py
│   │   └── make_dataset.py
│   │
│   ├── models           <- Model implementations, GAN training implementation
│   │   ├── autoencoder.py     <- GAN training loop containing training step, validation step, and loss functions
│   │   ├── generator.py       <- Generator model
│   │   ├── discriminator.py   <- Discriminator model
│   │   ├── DPRNN.py           <- Dual-path block in the discriminator model
│   │
│   ├── visualization    <- Scripts to create exploratory and results oriented visualizations
│   │   ├── __init__.py
│   │   └── visualize.py
│   ├── train_model.py   <- script for training the model
│   └── predict_model.py <- script for predicting from a model
│
├── models               <- Trained and serialized models, model predictions, or model summaries
│
├── reports              <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures          <- Generated graphics and figures to be used in reporting
│
├── Makefile             <- Makefile with convenience commands like `make data` or `make train`
│
├── README.md            <- The top-level README for developers using this project.
│
├── requirements.txt     <- The requirements file for reproducing the analysis environment
|
└── requirements_dev.txt <- The requirements file for reproducing the analysis environment
```


