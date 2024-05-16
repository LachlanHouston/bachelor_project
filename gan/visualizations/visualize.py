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

    # Visualize the mean of the feature maps
    plt.figure(figsize=(10, 10))
    plt.imshow(feature_maps_mean, cmap='viridis')
    plt.axis('off')
    plt.title('Mean of the feature maps of layer ' + str(layer))
    plt.savefig('reports/figures/' + save_name + '_feature_maps_mean' + str(layer) + '.png')

class GuidedBackprop:
    def __init__(self, model_path):
        self.generator = Autoencoder.load_from_checkpoint(model_path).generator
        self.discriminator = Autoencoder.load_from_checkpoint(model_path).discriminator
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def backward_hook(module, grad_in, grad_out):
            # Clamp the gradient of the output to follow a PRELU activation function
            if isinstance(grad_in[0], torch.Tensor):
                new_grad_in = torch.clamp(grad_in[0], min=0.0)
                return (new_grad_in,) + grad_in[1:]

        for module in self.generator.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d) or isinstance(module, nn.Linear) or isinstance(module, nn.LeakyReLU) or isinstance(module, nn.LSTM) or isinstance(module, nn.GroupNorm) or isinstance(module, nn.PReLU):
                self.hooks.append(module.register_backward_hook(backward_hook))

    def _get_discriminator_loss(self, real_clean, fake_clean, D_real, D_fake_no_grad):
        # Create interpolated samples
        alpha = torch.rand(1, 1, 1, 1, device='cpu') # B x 1 x 1 x 1
        # alpha = alpha.expand(real_clean.size()) # B x C x H x W
        differences = fake_clean - real_clean # B x C x H x W
        interpolates = real_clean + (alpha * differences) # B x C x H x W
        interpolates.requires_grad_(True)

        # Calculate the output of the discriminator for the interpolated samples and compute the gradients
        D_interpolates = self.discriminator(interpolates) # B x 1 (the output of the discriminator is a scalar value for each input sample)
        ones = torch.ones(D_interpolates.size(), device='cpu') # B x 1
        gradients = torch.autograd.grad(outputs=D_interpolates, inputs=interpolates, grad_outputs=ones, 
                                        create_graph=True, retain_graph=True)[0] # B x C x H x W
        
        # Calculate the gradient penalty
        gradients = gradients.view(1, -1) # B x (C*H*W)
        grad_norms = gradients.norm(2, dim=1) # B
        gradient_penalty = ((grad_norms - 1) ** 2).mean()

        # Adversarial loss
        D_adv_loss = D_fake_no_grad.mean() - D_real.mean()

        # Total discriminator loss
        D_loss = 10 * gradient_penalty + D_adv_loss

        return D_loss, 10 * gradient_penalty, D_adv_loss, None
    
    def visualize(self, input_tensor, target_tensor, discriminator):
        # Ensure input_tensor requires gradients
        input_tensor.requires_grad_(True)

        # Forward pass
        fake_clean = self.generator(input_tensor)[0]
        fake_clean = torch.tensor(fake_clean, requires_grad=True)

        # Fidelity loss
        fidelity = torch.norm(fake_clean - input_tensor, p=1) * 10

        # Discriminator output
        discriminator_output = self.discriminator(fake_clean)
        #discriminator_target = discriminator(target_tensor)

        #D_loss, penalty, D_adv_loss, L2_penalty = self._get_discriminator_loss(target_tensor, fake_clean, discriminator_target, discriminator_output)

        # Total loss
        loss = fidelity + (-discriminator_output.mean())
        
        # Zero gradients
        self.generator.zero_grad()
        
        # Backward pass from the loss
        loss.backward()
        
        # Gradient with respect to input
        return input_tensor.grad

    # def __del__(self):
    #     # Remove all hooks during cleanup
    #     for hook in self.hooks:
    #         hook.remove()

def plot_mask(filename, savename):
    # Load data and model
    (clean_waveform, noisy_waveform), (clean_stft, noisy_stft), (generator, discriminator) = get_data_and_model(filename, 'models/standardmodel1000.ckpt')

    output, mask = generator(noisy_stft)

    # Plot the mask with original input on the left and the output on the right
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot the original input
    mel_spec_noisy = librosa.feature.melspectrogram(y=noisy_waveform[0].numpy(), sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_noisy = librosa.power_to_db(mel_spec_noisy, ref=np.max)
    librosa.display.specshow(mel_spec_db_noisy, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[0])
    ax[0].set_title('Noisy Input')
    ax[0].set_xlabel('Time (s)')
    ax[0].set_ylabel('Frequency (Hz)')

    # Plot the mask
    mel_spec_mask = librosa.feature.melspectrogram(y=mask[0], sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_mask = librosa.power_to_db(mel_spec_mask, ref=np.max)
    librosa.display.specshow(mel_spec_db_mask, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[1])
    ax[1].set_title('Mask')
    ax[1].set_xlabel('Time (s)')
    ax[1].yaxis.set_visible(False)

    # Plot the output
    mel_spec_output = librosa.feature.melspectrogram(y=output[0], sr=16000, n_fft=512, hop_length=100, power=2, n_mels=64, fmax=8000)
    mel_spec_db_output = librosa.power_to_db(mel_spec_output, ref=np.max)
    librosa.display.specshow(mel_spec_db_output, y_axis='mel', x_axis='time', hop_length=100, sr=16000, ax=ax[2])
    ax[2].set_title('Fake Clean Output')
    ax[2].set_xlabel('Time (s)')
    ax[2].yaxis.set_visible(False)

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(ax[2].collections[0], cax=cbar_ax, use_gridspec=True, label='dB')

    plt.suptitle('Generated Mask from standardmodel')

    plt.savefig('reports/figures/' + savename + '_mask.png')
    plt.show()

def plot_validation_score(data, score_name, title, savename):
    score = data[score_name]
    epochs = data['epoch']
    # Plots the validation score
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, score)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.savefig('reports/figures/' + savename + '_validation_score.png')
    plt.show()

if __name__ == '__main__':
    print("Visualizing")

    # Create a GuidedBackprop object
    guided_backprop = GuidedBackprop('models/epoch=944.ckpt')

    # Load the input waveform
    input_waveform, sr = torchaudio.load('data/test_noisy_sampled/p232_012.wav')

    # Resample to 16kHz
    input = torchaudio.transforms.Resample(sr, 16000)(input_waveform)

    # Transform to STFT
    input = torch.stft(input, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
    input = torch.stack([input.real, input.imag], dim=1)

    # Visualize the guided backpropagation
    guided_backprop.visualize(input, None, None)

    # Plot the guided backpropagation as a spectrogram
    plt.figure(figsize=(10, 5))
    librosa.display.specshow(guided_backprop.visualize(input, None, None).squeeze(0).detach().cpu().numpy(), y_axis='mel', x_axis='time', hop_length=100, sr=16000)
    plt.colorbar()
    plt.title('Guided Backpropagation')
    plt.savefig('reports/figures/guided_backpropagation.png')
    plt.show()


