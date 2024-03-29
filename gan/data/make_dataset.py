import torchaudio
import os
import torch
from torch.nn import functional as F
from torchmetrics import ScaleInvariantSignalNoiseRatio

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

    return waveforms, sample_rates, files

def save_data(waveforms, path, filenames):
    """Save the data into the path"""
    for i, waveform in enumerate(waveforms):
        torch.save(waveform, path + filenames[i] + '.pt')

    print('Saved {} files'.format(len(waveforms)))

def process_data(waveforms, sample_rates, file_names, n_seconds=2):
    # Cut the waveforms to n seconds chunks
    new_waveforms = []
    new_file_names = []

    # Cut the waveforms to n seconds chunks, pad the last one if necessary
    for i, waveform in enumerate(waveforms):
        # Downsample the waveform to 16kHz
        new_freq=16000
        waveform = torchaudio.transforms.Resample(orig_freq=sample_rates[i], new_freq=new_freq)(waveform)

        # Cut the waveform to n seconds chunks and pad the last one if necessary. Save the chunks with save_data
        n_samples = n_seconds * new_freq
        n_chunks = waveform.size(1) // n_samples
        for j in range(n_chunks):
            new_waveforms.append(waveform[:, j*n_samples:(j+1)*n_samples])


            # Save the file names with the number of chunks, put the chunk number in the middle of the file name
            new_file_names.append(file_names[i][:8] + '_{}'.format(j))
        if waveform.size(1) % n_samples != 0:
            new_waveforms.append(F.pad(waveform[:, n_chunks*n_samples:], (0, n_samples - waveform.size(1) % n_samples)))
            new_file_names.append(file_names[i][:8] + '_{}'.format(n_chunks))
        
    print('Processed {} files'.format(len(new_waveforms)))

    return new_waveforms, new_file_names

def stft(new_waveforms):
    # Perform STFT on all the chunks
    stft_waveforms = []
    for i, waveform in enumerate(new_waveforms):
        stft_waveform = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        stacked = torch.stack((stft_waveform.real, stft_waveform.imag), dim=1)
        stft_waveforms.append(stacked)

    print('Processed {} files'.format(len(stft_waveforms)))
    return stft_waveforms

def create_simple_baseline(clean_raw, noisy_raw, test_clean_raw, test_noisy_raw):
    # Create a simple baseline
    # The simple baseline is find the mean mask between the clean and noisy data, and apply it to the noisy data
    # The mean mask is the mean of the clean data - noisy data
    # The mask is applied to the waveforms

    # Create the mean mask
    mean_mask = torch.zeros(clean_raw[0].size())
    for i in range(len(clean_raw)):
        mean_mask += clean_raw[i] - noisy_raw[i]
    mean_mask /= len(clean_raw)

    # Apply the mask to the test noisy data
    for i in range(len(test_noisy_raw)):
        test_noisy_raw[i] += mean_mask

    # Compute mean SNR
    snr = ScaleInvariantSignalNoiseRatio()
    snr_val = snr(test_clean_raw, test_noisy_raw)
    print('Mean SNR:', snr_val)

if __name__ == '__main__':
    print('Processing clean and noisy data...')
    clean_data_path = 'C:/Users/Lachl/Documents/GitHub/bachelor_project/org_data/clean_raw/'
    noisy_data_path = 'C:/Users/Lachl/Documents/GitHub/bachelor_project/org_data/noisy_raw/'

    # clean_waveforms, clean_sample_rates, clean_filenames = get_data(clean_data_path)
    # noisy_waveforms, noisy_sample_rates, noisy_filenames = get_data(noisy_data_path)
    test_clean_waveforms, test_clean_sample_rates, clean_test_filenames = get_data('org_data/test_clean_raw/')
    # test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames = get_data('org_data/test_noisy_raw/')

    


    # clean_waveforms, clean_filenames = process_data(clean_waveforms, clean_sample_rates, clean_filenames)
    # noisy_waveforms, noisy_filenames = process_data(noisy_waveforms, noisy_sample_rates, noisy_filenames)
    test_clean_waveforms, clean_test_filenames = process_data(test_clean_waveforms, test_clean_sample_rates, clean_test_filenames)
    # test_noisy_waveforms, noisy_test_filenames = process_data(test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames)


    # clean_waveforms = stft(clean_waveforms)
    # noisy_waveforms = stft(noisy_waveforms)
    test_clean_waveforms = stft(test_clean_waveforms)
    # test_noisy_waveforms = stft(test_noisy_waveforms)


    # assert_data(clean_waveforms)
    # assert_data(noisy_waveforms)
    # assert_data(test_clean_waveforms)
    # assert_data(test_noisy_waveforms)

    # save_data(clean_waveforms, 'C:/Users/Lachl/Documents/GitHub/bachelor_project/data/clean_stft/', clean_filenames)
    # save_data(noisy_waveforms, 'C:/Users/Lachl/Documents/GitHub/bachelor_project/data/noisy_stft/', noisy_filenames)
    save_data(test_clean_waveforms, 'data/test_clean_stft/', clean_test_filenames)
    # save_data(test_noisy_waveforms, 'data/test_noisy_stft/', noisy_test_filenames)

    # # load the processed data
    test = torch.load('data/test_noisy_stft/p232_001_0.pt')
    print(test.size())
