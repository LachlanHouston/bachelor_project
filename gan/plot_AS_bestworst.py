import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchaudio
import os
import librosa
import librosa.display
from gan.models.autoencoder import Autoencoder
from utils.utils import stft_to_waveform, compute_scores


def get_data_and_model(data_file_name, model_path):
    clean_file = 'data/test_clean_sampled/' + data_file_name
    noisy_file = 'data/test_noisy_sampled/' + data_file_name

    clean_waveform, clean_sample_rate = torchaudio.load(clean_file)
    noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_file)

    # Resample to 16kHz
    clean_waveform = torchaudio.transforms.Resample(clean_sample_rate, 16000)(clean_waveform)
    noisy_waveform = torchaudio.transforms.Resample(noisy_sample_rate, 16000)(noisy_waveform)

    # Transform to STFT
    clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    clean_stft = torch.stack([clean_stft.real, clean_stft.imag], dim=1)

    noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    noisy_stft = torch.stack([noisy_stft.real, noisy_stft.imag], dim=1)

    autoencoder = Autoencoder.load_from_checkpoint(model_path, return_waveform=True)
    generator = autoencoder.generator
    discriminator = autoencoder.discriminator

    return (clean_waveform, noisy_waveform), (clean_stft, noisy_stft), (generator, discriminator)

def plot_mask(filename, savename):
    # Load data and model
    if savename == 'standard':
        (clean_waveform, noisy_waveform), (clean_stft, noisy_stft), (generator, discriminator) = get_data_and_model(filename, 'models/final_standard_model945.ckpt')
    elif savename == 'supervised':
        (clean_waveform, noisy_waveform), (clean_stft, noisy_stft), (generator, discriminator) = get_data_and_model(filename, 'models/final_sisnr_loss_epoch=876.ckpt')

    output, mask = generator(noisy_stft)
    output = stft_to_waveform(output, device='cpu')
    mask = stft_to_waveform(mask, device='cpu')
    # Plot the mask with original input on the left and the output on the right
    fig, ax = plt.subplots(1, 4, figsize=(15, 15/4))
    
    fake_clean_waveform = output
    from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
    sisnr = ScaleInvariantSignalNoiseRatio()
    sisnr_score = sisnr(preds=fake_clean_waveform, target=clean_waveform).item()

    from torchaudio.pipelines import SQUIM_SUBJECTIVE
    non_matching_reference_waveform = torchaudio.load('data/test_clean_sampled/p232_002.wav')[0]
    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos_squim_score = subjective_model(fake_clean_waveform, non_matching_reference_waveform).item()
    
    print(savename+":\nSI-SNR: ", sisnr_score, "\nMOS Squim: ", mos_squim_score)

    # Plot the original input
    mel_spec_noisy = librosa.feature.melspectrogram(y=noisy_waveform[0].numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_noisy = librosa.power_to_db(mel_spec_noisy, ref=np.max)
    librosa.display.specshow(mel_spec_db_noisy, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[0])
    ax[0].set_title('Noisy Input')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')

    # Plot the mask
    mel_spec_mask = librosa.feature.melspectrogram(y=mask[0].detach().numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_mask = librosa.power_to_db(mel_spec_mask, ref=np.max)
    librosa.display.specshow(mel_spec_db_mask, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[1])
    ax[1].set_title('Mask')
    ax[1].set_xlabel('Time (s)')
    ax[1].yaxis.set_visible(False)

    # Plot the output
    mel_spec_output = librosa.feature.melspectrogram(y=output[0].detach().numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_output = librosa.power_to_db(mel_spec_output, ref=np.max)
    librosa.display.specshow(mel_spec_db_output, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[2])
    ax[2].set_title('Fake Clean Output')
    ax[2].set_xlabel('Time (s)')
    ax[2].yaxis.set_visible(False)

    # Plot the real clean
    mel_spec_clean = librosa.feature.melspectrogram(y=clean_waveform[0].numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_clean = librosa.power_to_db(mel_spec_clean, ref=np.max)
    librosa.display.specshow(mel_spec_db_clean, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[3])
    ax[3].set_title('Real Clean')
    ax[3].set_xlabel('Time (s)')
    ax[3].yaxis.set_visible(False)

    # plt.suptitle('Supervised Model')
    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax[2].collections[0], cax=cbar_ax, use_gridspec=True, label='dB')

    plt.subplots_adjust(bottom=0.13)

    plt.savefig( savename + '_mask.png', dpi=300)
    # plt.show()

def plot_specs(path1, path2, path3, path4):
    # Load waveforms
    waveform1, sr1 = librosa.load(path1, sr=16000)
    waveform2, sr2 = librosa.load(path2, sr=16000)
    waveform3, sr3 = librosa.load(path3, sr=16000)
    waveform4, sr4 = librosa.load(path4, sr=16000)

    fig, ax = plt.subplots(1, 4, figsize=(20*0.7, 5*0.7))

    # Compute and plot mel spectrogram for waveform 1
    mel_spec1 = librosa.feature.melspectrogram(y=waveform1, sr=sr1, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db1 = librosa.power_to_db(mel_spec1, ref=np.max)
    librosa.display.specshow(mel_spec_db1, y_axis='mel', x_axis='time', hop_length=100, sr=sr1, ax=ax[0])
    ax[0].set_xlabel('Time (s)', fontsize=12)
    ax[0].set_ylabel('Frequency (Hz)', fontsize=14)

    # Compute and plot mel spectrogram for waveform 2
    mel_spec2 = librosa.feature.melspectrogram(y=waveform2, sr=sr2, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db2 = librosa.power_to_db(mel_spec2, ref=np.max)
    librosa.display.specshow(mel_spec_db2, y_axis='mel', x_axis='time', hop_length=100, sr=sr2, ax=ax[1])
    ax[1].set_xlabel('Time (s)', fontsize=12)
    ax[1].yaxis.set_visible(False)

    # Compute and plot mel spectrogram for waveform 3
    mel_spec3 = librosa.feature.melspectrogram(y=waveform3, sr=sr3, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db3 = librosa.power_to_db(mel_spec3, ref=np.max)
    librosa.display.specshow(mel_spec_db3, y_axis='mel', x_axis='time', hop_length=100, sr=sr3, ax=ax[2])
    ax[2].set_xlabel('Time (s)', fontsize=12)
    ax[2].yaxis.set_visible(False)

    # Compute and plot mel spectrogram for waveform 4
    mel_spec4 = librosa.feature.melspectrogram(y=waveform4, sr=sr4, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db4 = librosa.power_to_db(mel_spec4, ref=np.max)
    librosa.display.specshow(mel_spec_db4, y_axis='mel', x_axis='time', hop_length=100, sr=sr4, ax=ax[3])
    ax[3].set_xlabel('Time (s)', fontsize=12)
    ax[3].yaxis.set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax[0].collections[0], cax=cbar_ax, use_gridspec=True, label='dB')

    plt.subplots_adjust(bottom=0.13)
    plt.savefig('spec_high.png', dpi=300)
    # plt.show()

# Example usage
waveform_paths = [
    '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/highest_improvement/2YK5_y1LzHk.wav',
    '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/highest_improvement/UhXKXjckLyY.wav',
    '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/highest_improvement/iW4gMn9MWcE.wav',
    '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/highest_improvement/ptHxsI2bt5g.wav',
]

# waveform_paths = [
#     '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/lowest_improvement/lld_AJcucqI.wav',
#     '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/lowest_improvement/rCNaKCymgp8.wav',
#     '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/lowest_improvement/t72LUnKPCSs.wav',
#     '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/lowest_improvement/6AyumUbU3VM.wav',
# ]

plot_specs(*waveform_paths)

# plot_mask('p257_112.wav', 'supervised')