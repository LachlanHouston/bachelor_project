import torchaudio
import os
import torch
from torch.nn import functional as F
from torchmetrics import ScaleInvariantSignalNoiseRatio
from wave import open as open_wave
import pandas as pd
import numpy as np
import tqdm as tqdm

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


def process_data_no_pad(waveforms, sample_rates, file_names):
    clips = []
    new_file_names = []
    org_sample_rate = sample_rates[0]

    for waveform, file_name in zip(waveforms, file_names):
        
        # Downsample the waveform to 16kHz
        sample_rate=16000
        waveform = torchaudio.transforms.Resample(orig_freq=org_sample_rate, new_freq=sample_rate)(waveform)

        # Duration of the audio in seconds
        total_duration = waveform.shape[1] / sample_rate
        
        # Duration of each clip to extract
        clip_duration = 2 # seconds

        # Calculate the number of clips and the interval between their starting points
        num_clips = int(np.ceil(total_duration / clip_duration)) if total_duration > clip_duration else -1
        interval = (total_duration - clip_duration) / (num_clips - 1) if num_clips >= 1 else None

        for i in range(num_clips if num_clips >= 1 else 1):
            if num_clips == -1:
                remaining = clip_duration - total_duration
                clip = torch.cat( (waveform, waveform[:,:int(remaining * sample_rate)+1]), dim=1)
                if clip.shape[1] == 32001:
                    clip = clip[:,:32000]

            else:    
                start_sec = i * interval
                start_frame = int(start_sec * sample_rate)
                end_frame = start_frame + int(clip_duration * sample_rate)

                # Ensure end_frame does not exceed waveform length
                if end_frame > waveform.shape[1]:
                    end_frame = waveform.shape[1]
                    start_frame = max(0, end_frame - int(clip_duration * sample_rate))

                clip = waveform[:, start_frame:end_frame]
            
            if clip.shape[1] != 32000:
                print('Shape:', clip.shape)
            clips.append(clip)
            
            # Save the file names with the number of chunks, put the chunk number in the middle of the file name
            new_file_names.append(file_name[:8] + '_{}'.format(i))

    return clips, new_file_names

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

def create_csv(path):
    total = len(os.listdir(path))
    filenames = os.listdir(path)
    # Create a dataframe with shape (n_files, 2*257*321)
    df = np.zeros((total, 2*257*321))
    for i, file in enumerate(filenames):
        if file.endswith('.pt'):
            stft = torch.load(path + file)
            stft = stft.squeeze(0).numpy() # SHAPE: (2, 257, 321)
            # Shape into (1, 2*257*321)
            stft = stft.reshape(1, -1)
            stft = stft[0]
            stft = stft.tolist()
            df[i] = stft
            print('Processed {}/{}'.format(i+1, total))
            if i == 10:
                break

    # Save as npy file
    np.save('data.npy', df)

    


if __name__ == '__main__':

    print('Processing clean and noisy data...')
    clean_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_wav_org/'
    noisy_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_wav_org/'
    # clean_test_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_wav_org/'
    # noisy_test_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_noisy_wav_org/'

    clean_waveforms, clean_sample_rates, clean_filenames = get_data(clean_data_path)
    noisy_waveforms, noisy_sample_rates, noisy_filenames = get_data(noisy_data_path)
    # test_clean_waveforms, test_clean_sample_rates, clean_test_filenames = get_data(clean_test_data_path)
    # test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames = get_data(noisy_test_data_path)

    clean_waveforms, clean_filenames = process_data_no_pad(clean_waveforms, clean_sample_rates, clean_filenames)
    noisy_waveforms, noisy_filenames = process_data_no_pad(noisy_waveforms, noisy_sample_rates, noisy_filenames)
    # test_clean_waveforms, clean_test_filenames = process_data_no_pad(test_clean_waveforms, test_clean_sample_rates, clean_test_filenames)
    # test_noisy_waveforms, noisy_test_filenames = process_data_no_pad(test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames)

    for i, waveform in enumerate(clean_waveforms):
        torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_wav/'+clean_filenames[i]+'.wav', waveform, 16000)
    for i, waveform in enumerate(noisy_waveforms):
        torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_wav/'+noisy_filenames[i]+'.wav', noisy_waveforms[i], 16000)
    # for i, waveform in enumerate(test_clean_waveforms):
    #     torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_wav/'+clean_test_filenames[i]+'.wav', test_clean_waveforms[i], 16000)
    # for i, waveform in enumerate(test_noisy_waveforms):
    #     torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_noisy_wav/'+noisy_test_filenames[i]+'.wav', test_noisy_waveforms[i], 16000)



    clean_waveforms = stft(clean_waveforms)
    noisy_waveforms = stft(noisy_waveforms)
    # test_clean_waveforms = stft(test_clean_waveforms)
    # test_noisy_waveforms = stft(test_noisy_waveforms)

    save_data(clean_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_stft/', clean_filenames)
    save_data(noisy_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_stft/', noisy_filenames)
    # save_data(test_clean_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_stft/', clean_test_filenames)
    # save_data(test_noisy_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_noisy_stft/', noisy_test_filenames)

    # load the processed data
    test = torch.load('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_stft/p231_469_2.pt')
    print("shape of an stft file:", test.size())

    # clean_stft_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_stft/'
    # noisy_stft_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_stft/'

    # create_csv(clean_stft_path)
