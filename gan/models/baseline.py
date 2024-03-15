import os
import torch
import torchaudio
from gan import compute_scores
import random
import tqdm
import numpy as np
import csv


def baseline_model():
    # Load test data
    test_clean_path = 'data/wav/test_clean_wav/'
    test_noisy_path = 'data/wav/test_noisy_wav/'
    test_clean_filenames = [file for file in os.listdir(test_clean_path) if file.endswith('.wav')]
    test_noisy_filenames = [file for file in os.listdir(test_noisy_path) if file.endswith('.wav')]
    test_clean_waveforms = []
    test_noisy_waveforms = []
    for i in tqdm.tqdm(range(len(test_clean_filenames)), "load test"):
        test_clean_waveforms.append(torchaudio.load(test_clean_path + test_clean_filenames[i])[0].squeeze(0))
        test_noisy_waveforms.append(torchaudio.load(test_noisy_path + test_noisy_filenames[i])[0].squeeze(0))

    # Load training data
    train_clean_path = 'data/wav/clean_wav/'
    train_noisy_path = 'data/wav/noisy_wav/'
    train_clean_filenames = [file for file in os.listdir(train_clean_path) if file.endswith('.wav')]
    train_noisy_filenames = [file for file in os.listdir(train_noisy_path) if file.endswith('.wav')]
    train_clean_waveforms = []
    train_noisy_waveforms = []
    for i in tqdm.tqdm(range(len(train_clean_filenames)), "load train"):
        train_clean_waveforms.append(torchaudio.load(train_clean_path + train_clean_filenames[i])[0].squeeze(0))
        train_noisy_waveforms.append(torchaudio.load(train_noisy_path + train_noisy_filenames[i])[0].squeeze(0))

    # Create the mean mask based on training data
    mean_mask = torch.zeros(train_clean_waveforms[0].size())
    for i in tqdm.tqdm(range(len(train_clean_waveforms)), "compute mask"):
        mean_mask += train_clean_waveforms[i] - train_noisy_waveforms[i]
    mean_mask /= len(train_clean_waveforms)

    # Apply the mask to the test noisy data
    fake_clean_test_waveforms = test_noisy_waveforms
    for i in range(len(fake_clean_test_waveforms)):
        fake_clean_test_waveforms[i] += mean_mask

    # Save the waveforms
    # for i in range(len(fake_clean_test_waveforms)):
    #     torchaudio.save('data/test_baseline/' + test_noisy_filenames[i], fake_clean_test_waveforms[i], 16000)

    # Turn into torch tensors
    test_clean_waveforms = torch.stack(test_clean_waveforms)
    fake_clean_test_waveforms = torch.stack(fake_clean_test_waveforms)

    ### Compute scores
    all_rows = []

    with open('baseline_scores.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch"])

        for i in tqdm.tqdm(range(len(test_clean_waveforms))):

            reference_index = random.choice([j for j in range(len(test_clean_waveforms)) if j != i])
            non_matching_reference_waveform = test_clean_waveforms[reference_index]
            sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score = compute_scores(
                                                test_clean_waveforms[i], fake_clean_test_waveforms[i], non_matching_reference_waveform)

            all_rows.append([sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score])
        
        ## Means
        writer.writerow(["Mean scores"])
        writer.writerow(np.mean(all_rows, axis=0))

        ## Standard errors of the means
        writer.writerow(["SE of the means" ])
        se_values = np.std(all_rows, axis=0) / np.sqrt(len(all_rows))
        writer.writerow(se_values)
        
        ## All scores
        writer.writerow(["All Scores"])
        for row in all_rows:
            writer.writerow(row)

    
if __name__ == '__main__':
    baseline_model()

    # SNR: 8.3789
    # eSTOI: 0.6115
    # SegSNR: 4.7741
    # DNSMOS: 2.3418
