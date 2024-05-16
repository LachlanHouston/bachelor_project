import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde
import os
import tqdm
import torchaudio

noisy_path_vctk = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_raw/'
noisy_filenames_vctk = sorted([file for file in os.listdir(noisy_path_vctk) if file.endswith('.wav')])

waveform_lengths_vctk = []
for filename in tqdm.tqdm(noisy_filenames_vctk):
    waveform, sr = torchaudio.load(os.path.join(noisy_path_vctk, filename))
    waveform = waveform.squeeze()
    waveform_lengths_vctk.append(len(waveform)/sr)

noisy_path_AudioSet = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_raw/'
noisy_filenames_AudioSet = sorted([file for file in os.listdir(noisy_path_AudioSet) if file.endswith('.wav')])

waveform_lengths_AudioSet = []
for filename in tqdm.tqdm(noisy_filenames_AudioSet):
    waveform, sr = torchaudio.load(os.path.join(noisy_path_AudioSet, filename))
    waveform = waveform.squeeze()
    waveform_lengths_AudioSet.append(len(waveform)/sr)


counts, bins = np.histogram(waveform_lengths_vctk, bins=100)
plt.hist(waveform_lengths_vctk, bins=bins, alpha=0.7, label='VCTKD')

counts, bins = np.histogram(waveform_lengths_AudioSet, bins=50)
plt.hist(waveform_lengths_AudioSet, bins=bins, alpha=0.7, label='AudioSet')

plt.title('Length of the Waveforms')
plt.xlabel('Length (Seconds)')
plt.ylabel('Count')
plt.legend()
plt.savefig('waveform_lengths.png', dpi=300)
