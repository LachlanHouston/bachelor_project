import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import random


class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path, is_train, fraction=1.0, authentic=False):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.authentic = authentic
        if fraction < 1.0:
            if self.authentic:
                clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.wav')])
                noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

                num_noisy_files_to_sample = int(fraction * len(noisy_files))
                noisy_indices = random.sample(range(len(noisy_files)), num_noisy_files_to_sample)
                self.noisy_files = [noisy_files[i] for i in noisy_indices]
                
                num_clean_files_to_sample = int(fraction * len(clean_files))
                clean_indices = random.sample(range(len(clean_files)), num_clean_files_to_sample)
                self.clean_files = [clean_files[i] for i in clean_indices]

            else:
                clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.wav')])
                noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])
                assert len(clean_files) == len(noisy_files), "Mismatch in number of clean and noisy files"
                num_files_to_sample = int(fraction * len(clean_files))
                indices = random.sample(range(len(clean_files)), num_files_to_sample)
                self.clean_files = [clean_files[i] for i in indices]
                self.noisy_files = [noisy_files[i] for i in indices]

        else:
            self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.wav')])
            self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

    def __len__(self):
        if self.authentic:
            return len(self.noisy_files)
        else:
            return min(len(self.noisy_files), len(self.clean_files))
    
    def __getitem__(self, idx):
        if self.authentic:
            clean_idx = random.randint(0, len(self.clean_files)-1)
        else:
            clean_idx = idx
        # Get length of the audio files
        clean_num_frames = torchaudio.info(os.path.join(self.clean_path, self.clean_files[clean_idx])).num_frames
        noisy_num_frames = torchaudio.info(os.path.join(self.noisy_path, self.noisy_files[idx])).num_frames
        clean_sample_rate = torchaudio.info(os.path.join(self.clean_path, self.clean_files[idx])).sample_rate
        noisy_sample_rate = torchaudio.info(os.path.join(self.noisy_path, self.noisy_files[idx])).sample_rate
        new_sample_rate = 16000 

        # If the audio is less than 2 seconds
        if clean_num_frames < 2*clean_sample_rate:
            clean_num_frames = 2*clean_sample_rate
        if noisy_num_frames < 2*noisy_sample_rate:
            noisy_num_frames = 2*noisy_sample_rate

        # Sample 2 seconds of audio randomly
        if self.authentic:
            noisy_start_frame = random.randint(0, noisy_num_frames-2*noisy_sample_rate)
            noisy_waveform, _ = torchaudio.load(os.path.join(self.noisy_path, self.noisy_files[idx]), frame_offset=noisy_start_frame, num_frames=2*noisy_sample_rate, backend='soundfile')
            clean_start_frame = random.randint(0, clean_num_frames-2*clean_sample_rate)
            clean_waveform, _ = torchaudio.load(os.path.join(self.clean_path, self.clean_files[clean_idx]), frame_offset=clean_start_frame, num_frames=2*clean_sample_rate, backend='soundfile')
        else:
            start_frame = random.randint(0, clean_num_frames-2*clean_sample_rate)
            clean_waveform, _ = torchaudio.load(os.path.join(self.clean_path, self.clean_files[clean_idx]), frame_offset=start_frame, num_frames=2*clean_sample_rate, backend='soundfile')
            noisy_waveform, _ = torchaudio.load(os.path.join(self.noisy_path, self.noisy_files[idx]), frame_offset=start_frame, num_frames=2*noisy_sample_rate, backend='soundfile')

        # Downsample the audio to 16kHz
        clean_waveform = torchaudio.transforms.Resample(orig_freq=clean_sample_rate, new_freq=new_sample_rate)(clean_waveform)
        noisy_waveform = torchaudio.transforms.Resample(orig_freq=noisy_sample_rate, new_freq=new_sample_rate)(noisy_waveform)

        # If the audio is less than 2 seconds, pad it with the start until it is 2 seconds
        if clean_waveform.shape[1] < 2*new_sample_rate:
            clean_waveform = torch.cat((clean_waveform, clean_waveform[:,:2*new_sample_rate-clean_waveform.shape[1]]), dim=1)
        if noisy_waveform.shape[1] < 2*new_sample_rate:
            noisy_waveform = torch.cat((noisy_waveform, noisy_waveform[:,:2*new_sample_rate-noisy_waveform.shape[1]]), dim=1)

        # Compute the STFT of the audio
        clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

        # Stack the real and imaginary parts of the STFT
        clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
        noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)

        return clean_stft, noisy_stft


# Lightning DataModule
class AudioDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16, fraction=1.0, authentic=False):
        super(AudioDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fraction = fraction
        self.authentic = authentic
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = AudioDataset(self.clean_path, self.noisy_path, is_train=True, fraction=self.fraction, authentic=self.authentic)
            self.val_dataset = AudioDataset(self.test_clean_path, self.test_noisy_path, is_train=False, authentic=self.authentic)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    

class MixDataSet(Dataset):
    def __init__(self, clean_path, noisy_path_authentic, noisy_path_paired, is_train=None, fraction=1.0):
        super(MixDataSet, self).__init__()
        self.paired_dataset = AudioDataset(clean_path, noisy_path_paired, is_train, fraction, authentic=False)
        self.unpaired_dataset = AudioDataset(clean_path, noisy_path_authentic, is_train, fraction, authentic=True)
    
    def __len__(self):
        return min(len(self.paired_dataset), len(self.unpaired_dataset))
    
    def __getitem__(self, idx):
        paired_samples = self.paired_dataset.__get_item__(idx)
        unpaired_samples = self.unpaired_dataset.__get_item__(idx)
        return paired_samples, unpaired_samples


class MixDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path_authentic, noisy_path_paired, test_clean_path, test_noisy_path_authentic, test_noisy_path_paired, batch_size=16, num_workers=16, fraction=1.0):
        super(MixDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path_authentic = noisy_path_authentic
        self.noisy_path_paired = noisy_path_paired
        self.test_clean_path = test_clean_path
        self.test_noisy_path_authentic = test_noisy_path_authentic
        self.test_noisy_path_paired = test_noisy_path_paired
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fraction = fraction
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MixDataSet(self.clean_path, self.noisy_path_authentic, self.noisy_path_paired, is_train=True, fraction=self.fraction)
            self.val_dataset = MixDataSet(self.test_clean_path, self.test_noisy_path_authentic, self.test_noisy_path_paired, is_train=False, fraction=self.fraction)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)


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
    VCTK_clean_path = os.path.join(os.getcwd(), 'data/test_clean_raw/')
    VCTK_noisy_path = os.path.join(os.getcwd(), 'data/test_noisy_raw/')

    dataloader = AudioDataset(VCTK_clean_path, VCTK_noisy_path, is_train=True, fraction=0.2)
    data = DataLoader(dataloader, batch_size=4, shuffle=True, num_workers=1, persistent_workers=True, pin_memory=False, drop_last=True)
    for clean, noisy in data:
        print(clean.shape, noisy.shape)
        # Turn into waveform
        clean_real = clean[0][:, 0, :, :]
        clean_imag = clean[0][:, 1, :, :]

        noisy_real = noisy[0][:, 0, :, :]
        noisy_imag = noisy[0][:, 1, :, :]

        clean = torch.complex(clean_real, clean_imag)
        noisy = torch.complex(noisy_real, noisy_imag)

        clean_waveform = torch.istft(clean, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))
        noisy_waveform = torch.istft(noisy, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))

        # Save as wav file
        torchaudio.save('clean.wav', clean_waveform, 16000)
        torchaudio.save('noisy.wav', noisy_waveform, 16000)

        break

    VCTK_clean_path = os.path.join(os.getcwd(), 'data/clean_raw/')
    VCTK_noisy_path = os.path.join(os.getcwd(), 'data/noisy_raw/')

    dataloader = AudioDataset(VCTK_clean_path, VCTK_noisy_path, is_train=True, fraction=0.2)
    data = DataLoader(dataloader, batch_size=4, shuffle=True, num_workers=1, persistent_workers=True, pin_memory=True, drop_last=True)
    for clean, noisy in data:
        print(clean.shape, noisy.shape)
        # Turn into waveform
        clean_real = clean[0][:, 0, :, :]
        clean_imag = clean[0][:, 1, :, :]

        noisy_real = noisy[0][:, 0, :, :]
        noisy_imag = noisy[0][:, 1, :, :]

        clean = torch.complex(clean_real, clean_imag)
        noisy = torch.complex(noisy_real, noisy_imag)

        clean_waveform = torch.istft(clean, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))
        noisy_waveform = torch.istft(noisy, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))

        # Save as wav file
        torchaudio.save('tclean.wav', clean_waveform, 16000)
        torchaudio.save('tnoisy.wav', noisy_waveform, 16000)

        break
