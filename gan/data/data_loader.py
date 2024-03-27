import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import random

# VCTK + DEMAND dataset class
class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.pt')])
        self.noisy_path = noisy_path
        self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.pt')])

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean_stft = torch.load(os.path.join(self.clean_path, self.clean_files[idx]))
        noisy_stft = torch.load(os.path.join(self.noisy_path, self.noisy_files[idx]))
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
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.vctk_train = AudioDataset(self.clean_path, self.noisy_path)
            self.vctk_val = AudioDataset(self.test_clean_path, self.test_noisy_path)

    def train_dataloader(self):
        return DataLoader(self.vctk_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.vctk_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)



# FSD50K dataset class
class FSD50KDataset(Dataset):
    def __init__(self, clean_path, noisy_path):
        super(FSD50KDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.pt')])
        self.noisy_path = noisy_path
        self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.pt')])

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean_idx = random.randint(0, len(self.clean_files)-1)
        clean_stft = torch.load(os.path.join(self.clean_path, self.clean_files[clean_idx]))
        noisy_stft = torch.load(os.path.join(self.noisy_path, self.noisy_files[idx]))
        return clean_stft, noisy_stft

# Lightning DataModule
class FSD50KDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16):
        super(FSD50KDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers if torch.cuda.is_available() else 1
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.FSD50K_train = FSD50KDataset(self.clean_path, self.noisy_path)
            self.FSD50K_val = FSD50KDataset(self.test_clean_path, self.test_noisy_path)

    def train_dataloader(self):
        return DataLoader(self.FSD50K_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.FSD50K_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, drop_last=True)



# Dummy dataset class
class DummyDataset(Dataset):
    def __init__(self, mean_dif=0, mode='train'):
        super(DummyDataset, self).__init__()
        if mode == 'train':
            self.clean_files = [torch.normal(mean=0, std=1, size=(2, 257, 321)) for _ in range(100)]
            self.noisy_files = [torch.normal(mean=0+mean_dif, std=1, size=(2, 257, 321)) for _ in range(100)]
        elif mode == 'val':
            self.clean_files = [torch.normal(mean=0, std=1, size=(2, 257, 321)) for _ in range(10)]
            self.noisy_files = [torch.normal(mean=0+mean_dif, std=1, size=(2, 257, 321)) for _ in range(10)]

    def __len__(self):
        return len(self.noisy_files)
    
    def __getitem__(self, idx):
        clean_stft = self.clean_files[idx]
        noisy_stft = self.noisy_files[idx]
        return clean_stft, noisy_stft

# Lightning DataModule
class DummyDataModule(L.LightningDataModule):
    def __init__(self, batch_size=16, num_workers=16, mean_dif=0):
        super(DummyDataModule, self).__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers if torch.cuda.is_available() else 1
        self.mean_dif = mean_dif
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.dummy_train = DummyDataset(mean_dif=self.mean_dif, mode='train')
            self.dummy_val = DummyDataset(mean_dif=self.mean_dif, mode='val')

    def train_dataloader(self):
        return DataLoader(self.dummy_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.dummy_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, drop_last=True)


if __name__ == '__main__':
   pass