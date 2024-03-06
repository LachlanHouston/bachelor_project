import torch
torch.manual_seed(42)
import torchmetrics
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import wandb
import io


def perfect_shuffle(tensor):
    # Ensure the tensor is at least 2D
    if tensor.dim() < 2:
        raise ValueError("Tensor must be at least 2-dimensional")

    size = tensor.size(0)
    idx = torch.randperm(size)
    
    # Check if the shuffle is a perfect derangement
    while (idx == torch.arange(size)).any():
        idx = torch.randperm(size)

    return tensor[idx]


def stft_to_waveform(stft, device=torch.device('cuda')):
    if len(stft.shape) == 3:
        stft = stft.unsqueeze(0)
    # Separate the real and imaginary components
    stft_real = stft[:, 0, :, :]
    stft_imag = stft[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)
    # Perform inverse STFT to obtain the waveform
    waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device))
    return waveform



