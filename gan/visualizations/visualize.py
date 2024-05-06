import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import librosa
import librosa.display
import cv2
from gan import Autoencoder

def get_data(path, num_files=3):
    """Get the data from the path"""
    files = os.listdir(path)

    # Shuffle the files

    # Get the first num_files files
    files = files[4:5]
    print('Loaded', len(files), 'files')

    # Load the waveforms and sample rates
    waveforms = []
    sample_rates = []
    for file in files:
        waveform, sample_rate = torchaudio.load(path + file)
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
    print('Loaded', len(waveforms), 'waveforms')
    return waveforms, sample_rates

def mel_spectrogram(clean_waveforms, noisy_waveforms, clean_sample_rates, noisy_sample_rates, title, save_name):
    # Plot 6 mel spectrograms, 3 clean and 3 noisy using librosa
    fig, ax = plt.subplots(2, 3, figsize=(15, 10))
    for i in range(3):
        mel_clean = librosa.feature.melspectrogram(y=clean_waveforms[i].numpy(), sr=clean_sample_rates[i], n_fft=512, hop_length=100, power=1, win_length=400, window='hann', n_mels=64)
        mel_noisy = librosa.feature.melspectrogram(y=noisy_waveforms[i].numpy(), sr=noisy_sample_rates[i], n_fft=512, hop_length=100, power=1, win_length=400, window='hann', n_mels=64)
        mel_clean = librosa.power_to_db(mel_clean[0, :, :], ref=np.max)
        mel_noisy = librosa.power_to_db(mel_noisy[0, :, :], ref=np.max)

        if i == 0:
            ax[0, i].set_ylabel('Clean')
            ax[1, i].set_ylabel('Noisy')
            librosa.display.specshow(mel_clean, y_axis='mel', hop_length=100, sr=clean_sample_rates[i], ax=ax[0, i], fmax=8000)
            librosa.display.specshow(mel_noisy, y_axis='mel', x_axis='time', hop_length=100, sr=noisy_sample_rates[i], ax=ax[1, i], fmax=8000)
        else:
            librosa.display.specshow(mel_clean, hop_length=100, sr=clean_sample_rates[i], ax=ax[0, i], fmax=8000)
            librosa.display.specshow(mel_noisy, x_axis='time', hop_length=100, sr=noisy_sample_rates[i], ax=ax[1, i], fmax=8000)

        ax[0, i].set_title('Clean ' + str(i + 1))
        ax[1, i].set_title('Noisy ' + str(i + 1))
        
    # Place one colorbar at the right of the last subplot
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax[1, 2].collections[0], cax=cbar_ax, use_gridspec=True, label='dB')

    plt.suptitle(title)
    plt.savefig('reports/figures/' + save_name + '_mel_spectrogram.png')
    plt.show()


def plot_waveforms(clean_waveforms, clean_sample_rates, noisy_waveforms, noisy_sample_rates, title, save_name):
    # Plot the waveforms
    fig, ax = plt.subplots(3, 2, figsize=(15, 10))
    for i in range(3):
        ax[i, 0].plot(clean_waveforms[i].numpy())
        ax[i, 0].set_title('Clean ' + str(i + 1))
        ax[i, 0].set_xlabel('Time')
        ax[i, 0].set_ylabel('Amplitude')
        ax[i, 1].plot(noisy_waveforms[i].numpy())
        ax[i, 1].set_title('Noisy ' + str(i + 1))
        ax[i, 1].set_xlabel('Time')
        ax[i, 1].set_ylabel('Amplitude')
    plt.suptitle(title)
    plt.savefig('reports/figures/' + save_name + '_waveforms.png')
    plt.show()

def find_global_max(waveforms):
    """Find the global maximum of all the waveforms"""
    max_amplitude = 0
    for waveform in waveforms:
        if waveform.max() > max_amplitude:
            max_amplitude = waveform.max()
    return max_amplitude

def generator_plot_loss(g_losses, titles, save_name):
    """Plot the generator losses"""
    # Generate 3 plots
    fig, ax = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    for i in range(3):
        ax[i].plot(g_losses[i])
        ax[i].set_title(titles[i])
        ax[i].set_ylabel('Loss')

        if i == 2:
            ax[i].set_xlabel('Epoch')

    plt.savefig('reports/figures/' + save_name + '_generator_loss.png')
    plt.show()

def discriminator_plot_loss(d_losses, titles, save_name):
    """Plot the discriminator losses"""
    # Generate 4 plots
    fig, ax = plt.subplots(2, 2, figsize=(15, 10), sharex=True)
    for i in range(4):
        ax[i // 2, i % 2].plot(d_losses[i])
        ax[i // 2, i % 2].set_title(titles[i])

        if i >= 2:
            ax[i // 2, i % 2].set_xlabel('Epoch')
            
        # On top row, set y-axis label to Output, on bottom row, set y-axis label to Loss
        if i < 2:
            ax[i // 2, i % 2].set_ylabel('Output')
        else:
            ax[i // 2, i % 2].set_ylabel('Loss')

    plt.savefig('reports/figures/' + save_name + '_discriminator_loss.png')
    plt.show()

def plot_discriminator_outputs(csv_path, save_name):
    """Plot the discriminator outputs"""
    df = pd.read_csv(csv_path, header=0)
    print(df.head())
    d_clean_scores = df['D_clean']
    d_fake_scores = df['D_fake']
    d_noisy_scores = df['D_noisy']

    # Plot probability distributions of the discriminator outputs in the same plot
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    ax.hist(d_clean_scores, bins=100, alpha=0.5, label='VCTK Noisy', density=True, color='r')
    #ax.hist(d_fake_scores, bins=100, alpha=0.5, label='D_fake', density=True, color='b')
    ax.hist(d_noisy_scores, bins=100, alpha=0.5, label='Audioset', density=True, color='g')

    # Fit a normal distribution to the data
    from scipy.stats import norm
    mu, std = norm.fit(d_clean_scores)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, color='r')

    # mu, std = norm.fit(d_fake_scores)
    # xmin, xmax = plt.xlim()
    # x = np.linspace(xmin, xmax, 200)
    # p = norm.pdf(x, mu, std)
    # ax.plot(x, p, 'k', linewidth=2, color='b')
    
    mu, std = norm.fit(d_noisy_scores)
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 200)
    p = norm.pdf(x, mu, std)
    ax.plot(x, p, 'k', linewidth=2, color='g')


    ax.legend(fontsize=15)
    ax.set_title('Discriminator Outputs of real clean and noisy, and fake generated clean waveforms', fontsize=15)
    ax.set_xlabel('Output', fontsize=15)
    ax.set_ylabel('Frequency', fontsize=15)
    plt.savefig('reports/figures/' + save_name + '_discriminator_outputs.png')
    plt.show()



if __name__ == '__main__':
    plot_discriminator_outputs('discriminator_scores_authentic.csv', 'histogram_authentic')
    

