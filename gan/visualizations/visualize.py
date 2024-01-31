import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import scipy.stats as stats

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

def amplitude_density(waveform, sample_rate, title, save_name):
    """Plot the amplitude density"""
    mean = waveform.mean()
    std = waveform.std()

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.hist(waveform.t().numpy(), bins=100, density=True)

    # Plot the PDF as an line
    xmin, xmax = -0.3, 0.3
    x = np.linspace(xmin, xmax, 100)
    p = stats.norm.pdf(x, mean, std)
    ax.plot(x, p, 'k', linewidth=2, color='r')


    plt.xlabel('Amplitude')
    plt.ylabel('Density')
    plt.xlim(xmin, xmax)
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_amplitude_density.png')
    plt.show()


if __name__ == '__main__':
    filename = "p226_009"

    clean_path = 'data/clean_raw/' + str(filename) + '.wav'
    noisy_path = 'data/noisy_raw/' + str(filename) + '.wav'

    clean_waveform, clean_sample_rate = torchaudio.load(clean_path)
    noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path)

    # Plot the waveform
    # plot_waveform(clean_waveform, clean_sample_rate, 'Clean waveform', filename)
    # plot_waveform(noisy_waveform, noisy_sample_rate, 'Noisy waveform', filename)

    # Plot the amplitude density
    amplitude_density(clean_waveform, clean_sample_rate, 'Clean amplitude density', filename)
    amplitude_density(noisy_waveform, noisy_sample_rate, 'Noisy amplitude density', filename)

