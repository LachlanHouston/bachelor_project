import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio

def plot_spectrogram(spectrogram, sample_rate, title, save_name):
    """Plot the spectrogram"""
    fig, ax = plt.subplots(figsize=(12, 4))

    # Remove all the zero padding
    for i in range(spectrogram.shape[2]):
        if torch.sum(spectrogram[:,:,i]) == 0:
            spectrogram = spectrogram[:,:,:i]
            break

    im = ax.imshow(spectrogram.log2()[0,:,:].numpy(), aspect='auto', origin='lower',
            interpolation='none')
    plt.colorbar(im, ax=ax)
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_spectrogram.png')

    plt.show()

def plot_waveform(waveform, sample_rate, title, save_name):
    """Plot the waveform"""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(waveform.t().numpy())
    plt.xlabel('Time')
    plt.ylabel('Amplitude')
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_waveform.png')

    plt.show()

def plot_waveform_and_spectrogram(waveform, spectrogram, sample_rate, title, save_name):
    """Plot the waveform and spectrogram"""
    plot_waveform(waveform, sample_rate, title, save_name)
    plot_spectrogram(spectrogram, sample_rate, title, save_name)

def plot_waveform_and_spectrogram_from_path(path, title, save_name):
    """Plot the waveform and spectrogram from the path"""
    waveform, sample_rate, spectrogram = torch.load(path)
    plot_waveform_and_spectrogram(waveform, spectrogram, sample_rate, title, save_name)

if __name__ == '__main__':
    filename = 0
    path = 'data/clean_processed/' + str(filename) + '.pt'
    plot_waveform_and_spectrogram_from_path(path, 'Clean', 'Clean_' + str(filename))

    path = 'data/noisy_processed/' + str(filename) + '.pt'
    plot_waveform_and_spectrogram_from_path(path, 'Noisy', 'Noisy_' + str(filename))