import matplotlib.pyplot as plt
from gan.models.autoencoder import Autoencoder
from gan.models.discriminator import Discriminator
from gan.models.generator import Generator
import os
import torch
import torchaudio
import numpy as np
import csv
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from gan.utils.utils import stft_to_waveform

PYTORCH_CUDA_ALLOC_CONF=True

def SNR_scores(train=True, model=None):
    clean_train_path = os.path.join(os.getcwd(), 'data/clean_raw/')
    noisy_train_path = os.path.join(os.getcwd(), 'data/noisy_raw/')
    clean_test_path = os.path.join(os.getcwd(), 'data/test_clean_raw/')
    noisy_test_path = os.path.join(os.getcwd(), 'data/test_noisy_raw/')

    # clean_train_files = sorted(os.listdir(clean_train_path))
    # noisy_train_files = sorted(os.listdir(noisy_train_path))
    clean_test_files = sorted(os.listdir(clean_test_path))
    noisy_test_files = sorted(os.listdir(noisy_test_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if model is None:
        model = Autoencoder(alpha_penalty =         10,
                            alpha_fidelity =        10,

                            n_critic =              10,
                            use_bias =              True,
                            
                            d_learning_rate =       1e-4,
                            g_learning_rate =       1e-4,

                            weight_clip =           False,
                            weight_clip_value =     1,

                            visualize =             True,
                            logging_freq =          5,
                            log_all_scores =        False,
                            batch_size =            4,
                            L2_reg =                False,
                            sisnr_loss =            False,
                            val_fraction =          1.,
                            dataset =               "VCTK"
                        ).generator

    model.to(device)

    snr_scores = []

    print("Test data")
    for i in range(len(clean_test_files)):
        print(f"File {i}/{len(clean_test_files)}")
        print(f"File: {clean_test_files[i]}")

        # Load data
        clean_waveform, sr = torchaudio.load(os.path.join(clean_test_path, clean_test_files[i]))
        noisy_waveform, _ = torchaudio.load(os.path.join(noisy_test_path, noisy_test_files[i]))

        # Resample to 16kHz
        clean_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(clean_waveform).to(device)
        noisy_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(noisy_waveform).to(device)

        # Transform waveform to STFT
        clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)
        noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)

        clean_stft = torch.stack([clean_stft.real, clean_stft.imag], dim=1)
        noisy_stft = torch.stack([noisy_stft.real, noisy_stft.imag], dim=1)
        
        output_real, _ = model(clean_stft)
        output_fake, _ = model(noisy_stft)

        # Transform the output to waveform
        real_clean_waveform = stft_to_waveform(output_real, device=device)
        fake_clean_waveform = stft_to_waveform(output_fake, device=device)

        # Compute SI-SNR
        sisnr = ScaleInvariantSignalNoiseRatio().to(device)
        snr = sisnr(preds=fake_clean_waveform, target=real_clean_waveform)
        print(f"SNR: {snr}")

        snr_scores.append(snr)

        # Clean up
        del clean_waveform, noisy_waveform, clean_stft, noisy_stft, output_real, output_fake, real_clean_waveform, fake_clean_waveform

        # Clear cache
        torch.cuda.empty_cache()

    return snr_scores

if __name__ == '__main__':
    model_names = ['standardmodel.ckpt']#, '20_cpkt', '30_cpkt', '40_cpkt', '50_cpkt', '60_cpkt', '70_cpkt', '80_cpkt', '90_cpkt', '100_cpkt']

    train_snr = []
    test_snr = []

    # Loop through all models
    for model_name in model_names:
        cpkt_path = os.path.join('models', model_name)
        cpkt_path = os.path.join(os.getcwd(), cpkt_path)
        model = Autoencoder.load_from_checkpoint(cpkt_path).generator
        
        #SNR_train = SNR_scores(train=True, model=None)
        SNR_test = SNR_scores(train=False, model=model)

        #train_snr.append([model_name, np.mean(SNR_train), np.std(SNR_train)])   
        test_snr.append([model_name, np.mean(SNR_test), np.std(SNR_test)])

        print(f"Model {model_name}:")
        #print(f"Train SNR: {np.mean(SNR_train)} +- {np.std(SNR_train)}")
        print(f"Test SNR: {np.mean(SNR_test)} +- {np.std(SNR_test)}")

    # Save scores to csv
    with open('train_snr.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(train_snr)

    with open('test_snr.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(test_snr)




