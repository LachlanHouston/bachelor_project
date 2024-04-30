import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torchaudio
import os
import librosa
import librosa.display
from gan.models.autoencoder import Autoencoder

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

    plt.savefig('reports/figures/' + save_name + '_discriminator_loss.png')
    plt.show()

def visualize_feature_maps(model_path, input_path, save_name, layer=0):
    """Visualize the feature maps of the model"""
    # Load the model
    model = Autoencoder.load_from_checkpoint(model_path).generator
    model.eval()

    # Load the input waveform
    input_waveform, sr = torchaudio.load(input_path)

    # Resample to 16kHz
    input = torchaudio.transforms.Resample(sr, 16000)(input_waveform)

    # Transform to STFT
    input = torch.stft(input, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    input = torch.stack([input.real, input.imag], dim=1)

    output, _, maps = model(input)

    # Visualize all the feature maps in the layer
    # Shape is (channels, frequency, time)
    feature_maps = maps[layer].squeeze(0).detach().cpu().numpy()

    # Mean of the channel dimension
    feature_maps_mean = feature_maps.mean(axis=0)

    # Visualize the feature maps packed in a single plot square
    # Scale size of the figure depending on the number of channels
    col = np.ceil(np.sqrt(feature_maps.shape[0]))
    row = np.ceil(feature_maps.shape[0] / col)
    fig = plt.figure(figsize=(col * 2, row * 2))
    for i in range(feature_maps.shape[0]):
        # Normalize the feature maps
        feature_maps[i] = (feature_maps[i] - feature_maps[i].min()) / (feature_maps[i].max() - feature_maps[i].min())

        # Create a subplot for each channel
        if layer==0:
            # We have 32 channels in the first layer
            ax = fig.add_subplot(8, 4, i+1)

        if layer==1:
            # We have 64 channels in the second layer
            ax = fig.add_subplot(8, 8, i+1)

        if layer==2:
            # We have 128 channels in the third layer
            ax = fig.add_subplot(16, 8, i+1)

        if layer==3:
            # We have 128 channels in the fourth layer
            ax = fig.add_subplot(16, 8, i+1)

        if layer==4:
            # We have 64 channels in the fifth layer
            ax = fig.add_subplot(8, 8, i+1)

        if layer==5:
            # We have 32 channels in the sixth layer
            ax = fig.add_subplot(8, 4, i+1)

        if layer==6:
            # We have 2 channels in the seventh layer
            ax = fig.add_subplot(1, 2, i+1)

        ax.imshow(feature_maps[i], cmap='viridis')
        ax.axis('off')
        ax.set_title('Channel ' + str(i), fontsize=10)
    plt.suptitle('Feature maps of layer ' + str(layer))

    plt.savefig('reports/figures/' + save_name + '_feature_maps' + str(layer) + '.png')
    #plt.show()

    # Visualize the mean of the feature maps
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_maps_mean, cmap='viridis')
    plt.axis('off')
    plt.title('Mean of the feature maps of layer ' + str(layer))
    plt.savefig('reports/figures/' + save_name + '_feature_maps_mean' + str(layer) + '.png')
    #plt.show()


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

import torch.nn as nn
import torch.nn.functional as F
class GuidedBackprop:
    def __init__(self, model_path):
        self.generator = Autoencoder.load_from_checkpoint(model_path).generator
        self.discriminator = Autoencoder.load_from_checkpoint(model_path).discriminator
        self.prelu_weights = []
        for name, module in self.generator.named_modules():
            if isinstance(module, nn.PReLU):
                self.prelu_weights.append(module.weight)

        self.hooks = []
        self._register_hooks()
        self.i = 0

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            # Clamp the gradient of the output to follow a PRELU activation function
            if isinstance(grad_in[0], torch.Tensor):
                new_grad_in = F.prelu(grad_in[0], self.prelu_weights[self.i])
                self.i += 1
                return (new_grad_in,) + grad_in[1:]

        for module in self.generator.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
                self.hooks.append(module.register_backward_hook(backward_hook))
    
    def visualize(self, input_tensor, target_tensor, discriminator):
        # Ensure input_tensor requires gradients
        input_tensor.requires_grad_(True)

        # Forward pass
        output = self.generator(input_tensor)[0]

        # Discriminator output
        discriminator_output = self.discriminator(output)
        #discriminator_target = discriminator(target_tensor)

        loss_adv = -discriminator_output.mean()
        
        # Compute the loss as the mean squared error between the output and the target
        fidelity = torch.norm(output - input_tensor, p=1) / (target_tensor.size(1) * target_tensor.size(2) * target_tensor.size(3))

        # Total loss
        loss = loss_adv + 10*fidelity
        
        # Zero gradients
        self.generator.zero_grad()
        
        # Backward pass from the loss
        loss.backward()
        
        # Gradient with respect to input
        return input_tensor.grad

    def __del__(self):
        # Remove all hooks during cleanup
        for hook in self.hooks:
            hook.remove()



if __name__ == '__main__':
    print("Visualizing")

    # model = Autoencoder.load_from_checkpoint('models/standardmodel1000.ckpt').generator

    # # Load losses from the training stored as a csv file
    # g_adv_loss = pd.read_csv('reports/g_adv.csv', header=None, skiprows=1)[4]
    # g_l1_loss = pd.read_csv('reports/g_fidelity.csv', header=None, skiprows=1)[4]
    # g_loss = pd.read_csv('reports/g_loss.csv', header=None, skiprows=1)[4]

    # # plot the generator losses
    # generator_plot_loss([g_adv_loss, g_l1_loss, g_loss], ['Adversarial Loss', 'Fidelity Loss', 'Total Generator Loss'], 'generator')

    # d_fake_loss = pd.read_csv('reports/d_fake.csv', header=None, skiprows=1)[4]
    # d_real_loss = pd.read_csv('reports/d_real.csv', header=None, skiprows=1)[4]
    # d_penalty_loss = pd.read_csv('reports/d_penalty.csv', header=None, skiprows=1)[4]
    # d_loss = pd.read_csv('reports/d_loss.csv', header=None, skiprows=1)[4]

    # # plot the discriminator losses
    # discriminator_plot_loss([d_fake_loss, d_real_loss, d_penalty_loss, d_loss], ['Discriminator Output (fake)', 'Discriminator Output (real)', 'Penalty Loss', 'Total Discriminator Loss'], 'discriminator')

    # Visualize the feature maps
    # model_path = 'models/standardmodel1000.ckpt'
    # input_path = 'data/test_noisy_sampled/p232_012.wav'
    # target_path = 'data/test_clean_sampled/p232_012.wav'

    # input_waveform, sr = torchaudio.load(input_path)
    # target_waveform, _ = torchaudio.load(target_path)

    # # Resample to 16kHz
    # input = torchaudio.transforms.Resample(sr, 16000)(input_waveform)
    # target = torchaudio.transforms.Resample(sr, 16000)(target_waveform)

    # # Transform to STFT
    # input = torch.stft(input, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    # input = torch.stack([input.real, input.imag], dim=1)

    # target = torch.stft(target, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    # target = torch.stack([target.real, target.imag], dim=1)

    # generator = Autoencoder.load_from_checkpoint(model_path).generator
    # discriminator = Autoencoder.load_from_checkpoint(model_path).discriminator

    # # input = torch.normal(0, 1, (1, 2, 257, 321))
    # # target = torch.normal(0, 1, (1, 2, 257, 321))
    
    # guided_bp = GuidedBackprop(model_path)
    # result = guided_bp.visualize(input, target, discriminator)
    # result = result.squeeze(0).squeeze(0).detach().cpu()

    # # Transform to waveform
    # real = result[0, :, :]
    # imag = result[1, :, :]
    # complex_result = torch.complex(real, imag)
    # result = torch.istft(complex_result, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))

    # fake_clean = generator(input)[0]
    # fake_clean = fake_clean.squeeze(0).squeeze(0).detach().cpu()
    # real = fake_clean[0, :, :]
    # imag = fake_clean[1, :, :]
    # complex_result = torch.complex(real, imag)
    # fake_clean = torch.istft(complex_result, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))

    # plt.figure(figsize=(15, 5))

    # plt.subplot(1, 3, 1)
    # mel_spec_input = librosa.feature.melspectrogram(y=input_waveform.numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64)
    # mel_spec_db_input = librosa.power_to_db(mel_spec_input[0, :, :], ref=np.max)
    # librosa.display.specshow(mel_spec_db_input, y_axis='mel', x_axis='time', hop_length=100, sr=16000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Input spectrogram')
    # plt.xlabel('Time (s)')
    # plt.ylabel('Frequency (Hz)')

    # plt.subplot(1, 3, 2)
    # mel_spect_rc = librosa.feature.melspectrogram(y=result.numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64)
    # mel_spect_db_rc = librosa.power_to_db(mel_spect_rc, ref=np.max)
    # librosa.display.specshow(mel_spect_db_rc, y_axis='mel', x_axis='time', hop_length=100, sr=16000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('GDP Result')
    # plt.xlabel('Time (s)')
    # # Hide y-axis
    # plt.gca().axes.get_yaxis().set_visible(False)

    # plt.subplot(1, 3, 3)
    # mel_spec_target = librosa.feature.melspectrogram(y=fake_clean.numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64)
    # mel_spec_db_target = librosa.power_to_db(mel_spec_target, ref=np.max)
    # librosa.display.specshow(mel_spec_db_target, y_axis='mel', x_axis='time', hop_length=100, sr=16000)
    # plt.colorbar(format='%+2.0f dB')
    # plt.title('Generator Result')
    # plt.xlabel('Time (s)')
    # # Hide y-axis
    # plt.gca().axes.get_yaxis().set_visible(False)

    # plt.savefig('reports/figures/guided_backpropagation.png')
    # plt.show()


