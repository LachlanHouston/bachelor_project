import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import librosa
import librosa.display

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
    plt.show()


# clean_path = os.path.join('data/test_clean_raw/') # 0.5799 train # 0.5057 test
#     noisy_path = os.path.join('data/test_noisy_raw/') # 0.9724 train # 0.9826 test
    
#     # Load 1 clean and 1 noisy waveforms
#     clean_waveforms, clean_sample_rates = get_data(clean_path, num_files=1)
#     noisy_waveforms, noisy_sample_rates = get_data(noisy_path, num_files=1)
#     print('Clean sample rate:', clean_sample_rates)
#     print('Noisy sample rate:', noisy_sample_rates)

#     # Turn into numpy arrays
#     clean_waveforms = clean_waveforms[0].numpy()
#     noisy_waveforms = noisy_waveforms[0].numpy()

#     # Transform the waveforms to mel spectrograms
#     mel_clean = librosa.feature.melspectrogram(y=clean_waveforms, sr=clean_sample_rates[0], n_fft=512, hop_length=100, power=2, win_length=400, window='hann', n_mels=64)
#     mel_noisy = librosa.feature.melspectrogram(y=noisy_waveforms, sr=noisy_sample_rates[0], n_fft=512, hop_length=100, power=2, win_length=400, window='hann', n_mels=64)
#     mel_clean = librosa.power_to_db(mel_clean[0, :, :], ref=np.max)
#     mel_noisy = librosa.power_to_db(mel_noisy[0, :, :], ref=np.max)

#     # Plot the waveforms and the corresponding mel spectrograms underneath
#     fig, ax = plt.subplots(2, 2, figsize=(15, 10))
#     ax[0, 0].plot(clean_waveforms[0])
#     ax[0, 0].xaxis.set_visible(False)
#     ax[0, 0].set_title('Clean')
#     ax[0, 0].set_ylabel('Amplitude')
#     ax[1, 0].plot(noisy_waveforms[0])
#     ax[1, 0].set_title('Noisy')
#     ax[1, 0].set_xlabel('Samples')
#     ax[1, 0].set_ylabel('Amplitude')
#     librosa.display.specshow(mel_clean, y_axis='mel', hop_length=100, sr=clean_sample_rates[0], ax=ax[0, 1], fmax=8000)
#     librosa.display.specshow(mel_noisy, y_axis='mel', x_axis='time', hop_length=100, sr=noisy_sample_rates[0], ax=ax[1, 1], fmax=8000)
#     ax[0, 1].set_title('Mel spectrogram of Clean')
#     # Move y-axis and unit to the right
#     ax[0, 1].yaxis.tick_right()
#     ax[0, 1].yaxis.set_label_position('right')
#     ax[1, 1].set_title('Mel spectrogram of Noisy')
#     # Move y-axis to the right
#     ax[1, 1].yaxis.tick_right()
#     ax[1, 1].yaxis.set_label_position('right')
#     plt.savefig('reports/figures/clean_noisy_waveforms_mel_spectrogram.png')

#     plt.show()



if __name__ == '__main__':

    # Load losses from the training stored as a csv file
    g_adv_loss = pd.read_csv('reports/g_adv.csv', header=None, skiprows=1)[4]
    g_l1_loss = pd.read_csv('reports/g_fidelity.csv', header=None, skiprows=1)[4]
    g_loss = pd.read_csv('reports/g_loss.csv', header=None, skiprows=1)[4]

    # plot the generator losses
    generator_plot_loss([g_adv_loss, g_l1_loss, g_loss], ['Adversarial Loss', 'Fidelity Loss', 'Total Generator Loss'], 'generator')

    d_fake_loss = pd.read_csv('reports/d_fake.csv', header=None, skiprows=1)[4]
    d_real_loss = pd.read_csv('reports/d_real.csv', header=None, skiprows=1)[4]
    d_penalty_loss = pd.read_csv('reports/d_penalty.csv', header=None, skiprows=1)[4]
    d_loss = pd.read_csv('reports/d_loss.csv', header=None, skiprows=1)[4]

    # plot the discriminator losses
    discriminator_plot_loss([d_fake_loss, d_real_loss, d_penalty_loss, d_loss], ['Discriminator Output (fake)', 'Discriminator Output (real)', 'Penalty Loss', 'Total Discriminator Loss'], 'discriminator')



