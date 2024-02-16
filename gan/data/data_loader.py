import torchaudio
import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile

class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path, standardize=None,
                 new_sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = [file for file in os.listdir(clean_path) if file.endswith('.pt')]

        self.noisy_path = noisy_path
        self.noisy_files = [file for file in os.listdir(noisy_path) if file.endswith('.pt')]

        self.new_sample_rate = new_sample_rate
        self.standardize = standardize

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean_file = self.clean_files[idx]
        noisy_file = self.noisy_files[idx]

        clean_stft = torch.load(os.path.join(self.clean_path, clean_file))
        noisy_stft = torch.load(os.path.join(self.noisy_path, noisy_file))

        return clean_stft, noisy_stft
    
def collate_fn(batch):
    clean_waveforms, noisy_waveforms = zip(*batch)
    return clean_waveforms, noisy_waveforms

def data_loader(clean_path = 'data/clean_stft', noisy_path = 'data/noisy_stft', 
                test_clean_path = 'data/test_clean_stft', test_noisy_path = 'data/test_noisy_stft',
                batch_size=16, num_workers=4):
    
    train_dataset = AudioDataset(clean_path, noisy_path, standardize=True)
    val_dataset = AudioDataset(test_clean_path, test_noisy_path, standardize=True)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=True, drop_last=True)
    return train_loader, val_loader


if __name__ == '__main__':
    clean_processed_path = 'data/clean_stft/'
    noisy_processed_path = 'data/noisy_stft/'
    
    train_loader, val_loader = data_loader(clean_processed_path, noisy_processed_path, batch_size=4, num_workers=4)
    print('Train:', len(train_loader), 'Validation:', len(val_loader))
    for batch in train_loader:
        clean_waveforms, noisy_waveforms = batch
        print('Clean:', clean_waveforms[0].shape, 'Noisy:', noisy_waveforms[0].shape)
        break