import torchaudio
import os
import torch
from torch.nn import functional as F

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

def process_data(waveforms, sample_rates):
    """Process the data into spectrograms"""
    # Pad the waveforms to have the same length
    max_length = max([waveform.shape[1] for waveform in waveforms])
    waveforms = [F.pad(waveform, (0, max_length - waveform.shape[1])) for waveform in waveforms]

    # Process the data into spectrograms
    spectrograms = []
    for waveform, sample_rate in zip(waveforms, sample_rates):
        spectrogram = torchaudio.transforms.Spectrogram()(waveform)
        spectrograms.append(spectrogram)

    print('Processed {} spectrograms'.format(len(spectrograms)))

    return spectrograms

def save_data(waveform, sample_rate, spectrograms, path):
    """Save the data to the path"""
    for i, spectrogram in enumerate(spectrograms):
        torch.save((waveform[i], sample_rate[i], spectrogram), path + str(i) + '.pt')


if __name__ == '__main__':
    clean_data_path = 'data/clean_raw/'
    clean_processed_path = 'data/clean_processed/'

    waveforms, sample_rates = get_data(clean_data_path)
    spectrograms = process_data(waveforms, sample_rates)
    save_data(waveforms, sample_rates, spectrograms, clean_processed_path)

    noisy_data_path = 'data/noisy_raw/'
    noisy_processed_path = 'data/noisy_processed/'

    waveforms, sample_rates = get_data(noisy_data_path)
    spectrograms = process_data(waveforms, sample_rates)
    save_data(waveforms, sample_rates, spectrograms, noisy_processed_path)