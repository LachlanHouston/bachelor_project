import numpy as np
import matplotlib.pyplot as plt
import torch
import torchaudio
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import librosa
import librosa.display

def get_data(path, num_files=3):
    """Get the data from the path"""
    files = os.listdir(path)

    # Get the first num_files files
    files = files[:num_files]
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
    clean_path = os.path.join('org_data/clean_trainset_28spk_wav/') # 0.5799 train # 0.5057 test
    noisy_path = os.path.join('org_data/noisy_trainset_28spk_wav/') # 0.9724 train # 0.9826 test
    
    # Load 3 clean and 3 noisy waveforms
    clean_waveforms, clean_sample_rates = get_data(clean_path)
    noisy_waveforms, noisy_sample_rates = get_data(noisy_path)

    clean_waveforms = clean_waveforms[:3]
    noisy_waveforms = noisy_waveforms[:3]

    # Plot the mel spectrograms
    #mel_spectrogram(clean_waveforms, noisy_waveforms, clean_sample_rates, noisy_sample_rates, 'Mel Spectrograms', 'mel_spectrogram_p1')

    # Plot the waveforms
    plot_waveforms(clean_waveforms, clean_sample_rates, noisy_waveforms, noisy_sample_rates, 'Waveforms', 'waveforms')



