import torchaudio
import os
import torch
from torch.nn import functional as F
from torchmetrics import ScaleInvariantSignalNoiseRatio
from wave import open as open_wave
import pandas as pd
import numpy as np
import tqdm

def get_data(path):
    """Get the data from the path"""
    files = os.listdir(path)
    errors = 0
    # Load all .wav files
    waveforms = []
    sample_rates = []
    for file in files:
        if file.endswith('.wav'):
            try:
                waveform, sample_rate = torchaudio.load(path + file)
                waveforms.append(waveform)
                sample_rates.append(sample_rate)
            except:
                errors += 1
                print('Error loading file:', file, 'errors:', errors)
    
    print('Loaded {} files'.format(len(waveforms)))

    return waveforms, sample_rates, files

def save_data(waveforms, path, filenames):
    """Save the data into the path"""
    for i, waveform in tqdm.tqdm(enumerate(waveforms)):
        torch.save(waveform, path + filenames[i] + '.pt')

    print('Saved {} files'.format(len(waveforms)))


def process_data(waveforms, sample_rates, file_names):
    clips = []
    new_file_names = []
    org_sample_rate = sample_rates[0]
    wrong_length = 0

    for waveform, file_name in tqdm.tqdm(zip(waveforms, file_names), "chopping up clips"):
        empties = 0
        if waveform.numel() == 0:
            empties += 1
            print('Empty waveform:', file_name, 'empties:', empties)
            continue
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
                clip = waveform
                while remaining > 0:
                    clip = torch.cat((clip, waveform[:,:int(remaining * sample_rate)+1]), dim=1)
                    remaining = clip_duration - clip.shape[1] / sample_rate
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
                print('File:', file_name)
                wrong_length += 1
            clips.append(clip)
            
            # Save the file names with the number of chunks, put the chunk number in the middle of the file name
            new_file_names.append(file_name[:8] + '_{}'.format(i))
    print('Clips with wrong length:', wrong_length, 'out of', len(clips))

    clips = normalize_globally(clips)

    return clips, new_file_names

def normalize_globally(waveforms):
    """Normalize the waveforms globally"""
    max_value = 0
    for waveform in waveforms:
        if waveform.max() > max_value:
            max_value = waveform.max()
    print('Max value:', max_value)
    for i, waveform in enumerate(waveforms):
        waveforms[i] = waveform / max_value
    return waveforms

def stft(new_waveforms):
    # Perform STFT on all the chunks
    stft_waveforms = []
    for i, waveform in tqdm.tqdm(enumerate(new_waveforms)):
        stft_waveform = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        stacked = torch.stack((stft_waveform.real, stft_waveform.imag), dim=1)
        stft_waveforms.append(stacked)
    print('Processed {} files'.format(len(stft_waveforms)))
    return stft_waveforms

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
    # clean_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_wav_raw/'
    noisy_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_raw/'
    # clean_test_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_wav_raw/'
    noisy_test_data_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/test_raw/'

    # clean_waveforms, clean_sample_rates, clean_filenames = get_data(clean_data_path)
    noisy_waveforms, noisy_sample_rates, noisy_filenames = get_data(noisy_data_path)
    # test_clean_waveforms, test_clean_sample_rates, clean_test_filenames = get_data(clean_test_data_path)
    test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames = get_data(noisy_test_data_path)

    # clean_waveforms, clean_filenames = process_data(clean_waveforms, clean_sample_rates, clean_filenames)
    noisy_waveforms, noisy_filenames = process_data(noisy_waveforms, noisy_sample_rates, noisy_filenames)
    # test_clean_waveforms, clean_test_filenames = process_data(test_clean_waveforms, test_clean_sample_rates, clean_test_filenames)
    test_noisy_waveforms, noisy_test_filenames = process_data(test_noisy_waveforms, test_noisy_sample_rates, noisy_test_filenames)

    ## Save in waveform format
    # for i, waveform in enumerate(clean_waveforms):
        # torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_wav/'+clean_filenames[i]+'.wav', waveform, 16000)
    for i, waveform in tqdm.tqdm(enumerate(noisy_waveforms)):
        torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_wav2/'+noisy_filenames[i]+'.wav', noisy_waveforms[i], 16000)
    # for i, waveform in enumerate(test_clean_waveforms):
    #     torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_wav/'+clean_test_filenames[i]+'.wav', test_clean_waveforms[i], 16000)
    for i, waveform in tqdm.tqdm(enumerate(test_noisy_waveforms)):
        torchaudio.save('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/test_wav2/'+noisy_test_filenames[i]+'.wav', test_noisy_waveforms[i], 16000)



    # # clean_waveforms = stft(clean_waveforms)
    # noisy_waveforms = stft(noisy_waveforms)
    # # test_clean_waveforms = stft(test_clean_waveforms)
    # test_noisy_waveforms = stft(test_noisy_waveforms)

    # ## Save in STFT format
    # # save_data(clean_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_stft/', clean_filenames)
    # save_data(noisy_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_stft/', noisy_filenames)
    # # save_data(test_clean_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_stft/', clean_test_filenames)
    # save_data(test_noisy_waveforms, '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/test_stft/', noisy_test_filenames)




    # load the processed data
    # test = torch.load('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_stft/p231_469_2.pt')
    # print("shape of an stft file:", test.size())

    # clean_stft_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/clean_stft/'
    # noisy_stft_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_stft/'

    # create_csv(clean_stft_path)
