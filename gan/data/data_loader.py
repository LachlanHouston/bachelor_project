import os
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L

class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path, standardize=None,
                 new_sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.pt')])


        self.noisy_path = noisy_path
        self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.pt')])

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

# Lightning DataModule
class VCTKDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16):
        super(VCTKDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers if torch.cuda.is_available() else 1

        self.save_hyperparameters()


    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.vctk_train = AudioDataset(self.clean_path, self.noisy_path)
            self.vctk_val = AudioDataset(self.test_clean_path, self.test_noisy_path)
            

    def train_dataloader(self):
        return DataLoader(self.vctk_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.vctk_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, drop_last=True)


if __name__ == '__main__':
   pass