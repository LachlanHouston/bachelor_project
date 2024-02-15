import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def get_data(path):
    """Get the data from the path"""
    files = os.listdir(path)

    # Load all .wav files
    waveforms = []
    sample_rates = []
    for file in files:
        if file.endswith('.wav'):
            waveform, sample_rate = torchaudio.load(path + file)
            waveforms.append(waveform)
            sample_rates.append(sample_rate)
    
    print('Loaded {} files'.format(len(waveforms)))

    return waveforms, sample_rates

def stft_spectrogram(waveform, sample_rate, title, save_name):
    spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=100, win_length=400)(waveform)
    spectrogram = torch.squeeze(spectrogram, dim=0)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(spectrogram, aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_stft_spectrogram.png')
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

def spectral_envelope(waveform, sample_rate, title, save_name):
    """Plot the spectral envelope"""
    # Apply STFT
    Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True)
    Xstft_real = Xstft.real
    Xstft_imag = Xstft.imag
    Xstft = torch.stack([Xstft_real, Xstft_imag], dim=1)
    Xstft = torch.squeeze(Xstft, dim=0)

    # Get the magnitude of the complex-valued spectrogram
    Xmag = torch.sqrt(Xstft_real ** 2 + Xstft_imag ** 2)
    Xmag = torch.squeeze(Xmag, dim=0)

    # Get the phase of the complex-valued spectrogram
    Xphase = torch.atan2(Xstft_imag, Xstft_real)
    Xphase = torch.squeeze(Xphase, dim=0)

    # Plot the magnitude of the complex-valued spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(Xmag.t().numpy(), aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_spectral_envelope.png')
    plt.show()

    # Plot the phase of the complex-valued spectrogram
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.imshow(Xphase.t().numpy(), aspect='auto', origin='lower')
    plt.xlabel('Time')
    plt.ylabel('Frequency')
    plt.title(title)

    # Save the plot
    plt.savefig('reports/figures/' + save_name + '_spectral_envelope.png')
    plt.show()

def find_global_max(waveforms):
    """Find the global maximum of all the waveforms"""
    max_amplitude = 0
    for waveform in waveforms:
        if waveform.max() > max_amplitude:
            max_amplitude = waveform.max()
    return max_amplitude



if __name__ == '__main__':
    clean_path = os.path.join('data/test_clean_processed/') # 0.5799 train # 0.5057 test
    noisy_path = os.path.join('data/test_noisy_processed/') # 0.9724 train # 0.9826 test
    waveforms, sample_rates = get_data(clean_path)
    max_amplitude = find_global_max(waveforms)
    print(max_amplitude)

    # clean_waveform, clean_sample_rate = torchaudio.load(clean_path)
    # noisy_waveform, noisy_sample_rate = torchaudio.load(noisy_path)

    # Plot the waveform
    # plot_waveform(clean_waveform, clean_sample_rate, 'Clean waveform', filename)
    # plot_waveform(noisy_waveform, noisy_sample_rate, 'Noisy waveform', filename)

    # Plot the amplitude density
    # amplitude_density(clean_waveform, clean_sample_rate, 'Clean amplitude density', filename)
    # amplitude_density(noisy_waveform, noisy_sample_rate, 'Noisy amplitude density', filename)

    # Plot the STFT spectrogram
    # stft_spectrogram(clean_waveform, clean_sample_rate, 'Clean STFT spectrogram_clean', filename + '_clean')
    # stft_spectrogram(noisy_waveform, noisy_sample_rate, 'Noisy STFT spectrogram_noisy', filename + '_noisy')



