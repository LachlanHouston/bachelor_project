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
from gan.utils.utils import waveform_to_stft, stft_to_waveform

def SNR_scores(train=True, model=None, device='cpu'):
    clean_train_path = os.path.join(os.getcwd(), 'data/clean_raw/')
    noisy_train_path = os.path.join(os.getcwd(), 'data/noisy_raw/')
    clean_test_path = os.path.join(os.getcwd(), 'data/test_clean_raw/')
    noisy_test_path = os.path.join(os.getcwd(), 'data/test_noisy_raw/')

    clean_train_files = sorted(os.listdir(clean_train_path))
    noisy_train_files = sorted(os.listdir(noisy_train_path))
    clean_test_files = sorted(os.listdir(clean_test_path))
    noisy_test_files = sorted(os.listdir(noisy_test_path))

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

    snr_scores = []
    i = 0

    if train:
        print("Train data")
        for data in zip(clean_train_files, noisy_train_files):
            print(f"File {i}/{len(clean_train_files)}")

            # Load data
            clean_waveform, clean_sr = torchaudio.load(os.path.join(clean_train_path, data[0]))
            noisy_waveform, noisy_sr = torchaudio.load(os.path.join(noisy_train_path, data[1]))

            # Resample to 16kHz
            clean_waveform = torchaudio.transforms.Resample(orig_freq=clean_sr, new_freq=16000)(clean_waveform)
            noisy_waveform = torchaudio.transforms.Resample(orig_freq=noisy_sr, new_freq=16000)(noisy_waveform)

            # Transform waveform to STFT
            clean_stft = waveform_to_stft(clean_waveform, device=device)
            noisy_stft = waveform_to_stft(noisy_waveform, device=device)
            
            output_real, _ = model(clean_stft)
            output_fake, _ = model(noisy_stft)

            # Transform the output to waveform
            real_clean_waveform = stft_to_waveform(output_real, device=device)
            fake_clean_waveform = stft_to_waveform(output_fake, device=device)

            # Compute SI-SNR
            sisnr = ScaleInvariantSignalNoiseRatio()
            snr = sisnr(preds=fake_clean_waveform, target=real_clean_waveform).item()
            print(f"SNR: {snr}")

            snr_scores.append(snr)

            i += 1

    else:
        print("Test data")
        for data in zip(clean_test_files, noisy_test_files):
            print(f"File {i}/{len(clean_test_files)}")

            # Load data
            clean_waveform, _ = torchaudio.load(os.path.join(clean_test_path, data[0]))
            noisy_waveform, _ = torchaudio.load(os.path.join(noisy_test_path, data[1]))

            # Transform waveform to STFT
            clean_stft = waveform_to_stft(clean_waveform, device=device)
            noisy_stft = waveform_to_stft(noisy_waveform, device=device)
            
            output_real, _ = model(clean_stft)
            output_fake, _ = model(noisy_stft)

            # Transform the output to waveform
            real_clean_waveform = stft_to_waveform(output_real, device=device)
            fake_clean_waveform = stft_to_waveform(output_fake, device=device)

            # Compute SI-SNR
            sisnr = ScaleInvariantSignalNoiseRatio()
            snr = sisnr(preds=fake_clean_waveform, target=real_clean_waveform).item()
            print(f"SNR: {snr}")

            snr_scores.append(snr)

            i += 1

    return snr_scores

if __name__ == '__main__':
    model_names = ['standardmodel.ckpt']#, '20_cpkt', '30_cpkt', '40_cpkt', '50_cpkt', '60_cpkt', '70_cpkt', '80_cpkt', '90_cpkt', '100_cpkt']

    train_snr = []
    test_snr = []

    # Loop through all models
    for model_name in model_names:
        cpkt_path = os.path.join('models', model_name)
        cpkt_path = os.path.join(os.getcwd(), cpkt_path)
        model = Autoencoder.load_from_checkpoint(cpkt_path,  
                                                visualize=False,
                                                alpha_penalty=10,
                                                alpha_fidelity=10,
                                                n_critic=10,
                                                d_learning_rate=1e-4,
                                                d_scheduler_step_size=1000,
                                                d_scheduler_gamma=1,
                                                g_learning_rate=1e-4,
                                                g_scheduler_step_size=1000,
                                                g_scheduler_gamma=1,
                                                weight_clip = False,
                                                weight_clip_value=0.5,
                                                logging_freq=5,
                                                batch_size=16).generator
        
        #SNR_train = SNR_scores(train=True, model=None)
        SNR_test = SNR_scores(train=False, model=None)

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




