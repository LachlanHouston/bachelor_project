import os
import torch
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
import pytorch_lightning as L
from tqdm import tqdm

class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = [file for file in os.listdir(clean_path) if file.endswith('.pt')]

        self.noisy_path = noisy_path
        self.noisy_files = [file for file in os.listdir(noisy_path) if file.endswith('.pt')]

        # Load the data
        self.clean_data = torch.zeros(len(self.clean_files), 2, 257, 321)
        self.noisy_data = torch.zeros(len(self.noisy_files), 2, 257, 321)
        
        print("Loading data...")
        for i, file in enumerate(tqdm(self.noisy_files)):
            self.clean_data[i] = torch.load(os.path.join(self.clean_path, file))
            self.noisy_data[i] = torch.load(os.path.join(self.noisy_path, file))

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean_stft = self.clean_data[idx]
        noisy_stft = self.noisy_data[idx]

        return clean_stft, noisy_stft

# Lightning DataModule
class VCTKDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16):
        super(VCTKDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.vctk_train = AudioDataset(self.clean_path, self.noisy_path)
            self.vctk_val = AudioDataset(self.test_clean_path, self.test_noisy_path)
            

    def train_dataloader(self):
        return DataLoader(self.vctk_train, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.vctk_val, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=False, pin_memory=True, drop_last=True)


if __name__ == '__main__':
    pass