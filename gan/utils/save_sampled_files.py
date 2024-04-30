import torch
import torchaudio
from gan import Autoencoder
from gan import AudioDataset
from torch.utils.data import DataLoader
from gan import stft_to_waveform, compute_scores
from pytorch_lightning import Trainer
import os
import numpy as np
from tqdm import tqdm
import csv
import random
import numpy as np
import tqdm
torch.set_grad_enabled(False)
PYTORCH_ENABLE_MPS_FALLBACK=1

clean_path = 'data/test_clean_raw/'
noisy_path = 'data/test_noisy_raw/'
device = torch.device('mps')

if __name__ == '__main__':
    dataset = AudioDataset(clean_path=clean_path, noisy_path=noisy_path, is_train=False, authentic=False)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1 if torch.device == 'cpu' else 5, 
                            persistent_workers=True, pin_memory=True, drop_last=True)

    for i, batch in tqdm.tqdm(enumerate(data_loader)):
        real_clean = batch[0].squeeze(1)
        real_noisy = batch[1].squeeze(1)

        clean_filename = batch[2][0]
        noisy_filename = batch[3][0]

        clean_wav = stft_to_waveform(real_clean, device='cpu')
        noisy_wav = stft_to_waveform(real_noisy, device='cpu')

        torchaudio.save(f'/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_clean_sampled2/{clean_filename}', clean_wav, 16000)
        torchaudio.save(f'/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/test_noisy_sampled2/{noisy_filename}', noisy_wav, 16000)



