import os
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torch_pesq import PesqLoss


def baseline_model():
    # Load all test data
    clean_filenames = os.listdir('org_data/test_clean_raw/')
    noisy_filenames = os.listdir('org_data/test_noisy_raw/')

    clean_waveforms = []
    noisy_waveforms = []

    for i in range(len(clean_filenames)):
        clean_waveforms.append(torchaudio.load('org_data/test_clean_raw/' + clean_filenames[i])[0])
        noisy_waveforms.append(torchaudio.load('org_data/test_noisy_raw/' + noisy_filenames[i])[0])
    
    # Find largest waveform
    max_length = 0
    for waveform in clean_waveforms:
        if waveform.size(1) > max_length:
            max_length = waveform.size(1)

    # Pad all waveforms to the same length
    for i in range(len(clean_waveforms)):
        clean_waveforms[i] = torch.nn.functional.pad(clean_waveforms[i], (0, max_length - clean_waveforms[i].size(1)))
        noisy_waveforms[i] = torch.nn.functional.pad(noisy_waveforms[i], (0, max_length - noisy_waveforms[i].size(1)))

    # Create the mean mask
    mean_mask = torch.zeros(clean_waveforms[0].size())
    for i in range(len(clean_waveforms)):
        mean_mask += clean_waveforms[i] - noisy_waveforms[i]

    mean_mask /= len(clean_waveforms)

    # Apply the mask to the test noisy data
    for i in range(len(noisy_waveforms)):
        noisy_waveforms[i] += mean_mask

    # Save the waveforms
    # for i in range(len(noisy_waveforms)):
    #     torchaudio.save('data/test_baseline/' + noisy_filenames[i], noisy_waveforms[i], 16000)

    # Turn into torch tensors
    clean_waveforms = torch.stack(clean_waveforms)
    noisy_waveforms = torch.stack(noisy_waveforms)

    # Compute mean SNR
    snr = ScaleInvariantSignalNoiseRatio()
    snr_val = snr(clean_waveforms, noisy_waveforms)
    print('Mean SNR:', snr_val)
    
if __name__ == '__main__':
    baseline_model()

    # Mean SNR is 8.3789

