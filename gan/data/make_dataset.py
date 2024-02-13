#%%
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

    return waveforms, sample_rates, files

def save_data(waveforms, path, filenames):
    """Save the data into the path"""
    for i, waveform in enumerate(waveforms):
        torchaudio.save(path + filenames[i], waveform, 16000)

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
            new_file_names.append(file_names[i][:8] + '_{}'.format(j) + file_names[i][8:])
        if waveform.size(1) % n_samples != 0:
            new_waveforms.append(F.pad(waveform[:, n_chunks*n_samples:], (0, n_samples - waveform.size(1) % n_samples)))
            new_file_names.append(file_names[i][:8] + '_{}'.format(n_chunks) + file_names[i][8:])

    print('Processed {} files'.format(len(new_waveforms)))

    return new_waveforms, new_file_names

def assert_data(waveforms):
    """Assert that the data is correct"""
    for i, waveform in enumerate(waveforms):
        assert waveform.size(1) == 32000 or waveform.size(1) == 16000

if __name__ == '__main__':
    print('Processing clean and noisy data...')
    clean_data_path = 'data/clean_raw/'
    noisy_data_path = 'data/noisy_raw/'

    clean_waveforms, clean_sample_rates, clean_filenames = get_data(clean_data_path)
    # noisy_waveforms, noisy_sample_rates, noisy_filenames = get_data(noisy_data_path)
    # test_clean_waveforms, test_clean_sample_rates, clean_test_filenames = get_data('data/test_clean_raw/')
    # test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames = get_data('data/test_noisy_raw/')

#%%

    clean_waveforms, clean_filenames = process_data(clean_waveforms, clean_sample_rates, clean_filenames)
    # noisy_waveforms, noisy_filenames = process_data(noisy_waveforms, noisy_sample_rates, noisy_filenames)
    # test_clean_waveforms, clean_test_filenames = process_data(test_clean_waveforms, test_clean_sample_rates, clean_test_filenames)
    # test_noisy_waveforms, noisy_test_filenames = process_data(test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames)

    # print(len(clean_waveforms), len(clean_filenames))
    # print(clean_filenames[:10])

#%%

    assert_data(clean_waveforms)
    # assert_data(noisy_waveforms)
    # assert_data(test_clean_waveforms)
    # assert_data(test_noisy_waveforms)

    save_data(clean_waveforms, 'data/clean_processed/', clean_filenames)
    # save_data(noisy_waveforms, 'data/noisy_processed/', noisy_filenames)
    # save_data(test_clean_waveforms, 'data/test_clean_processed/', clean_test_filenames)
    # save_data(test_noisy_waveforms, 'data/test_noisy_processed/', noisy_test_filenames)