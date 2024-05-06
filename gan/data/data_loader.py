import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as L
import random


class PreMadeDataset(Dataset):
    def __init__(self, clean_path, noisy_path, fraction=1.):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        num_files = int(fraction * len(os.listdir(clean_path)))
        self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.wav')])[:num_files]
        self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])[:num_files]
    
    def __len__(self):
        return min(len(self.clean_files), len(self.noisy_files))
    
    def __getitem__(self, idx):
        clean_waveform, _ = torchaudio.load(os.path.join(self.clean_path, self.clean_files[idx]))
        noisy_waveform, _ = torchaudio.load(os.path.join(self.noisy_path, self.noisy_files[idx]))
        # Compute the STFT of the audio
        clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

        # Stack the real and imaginary parts of the STFT
        clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
        noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)
        return clean_stft, noisy_stft
    

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

        return clean_stft.squeeze(), noisy_stft.squeeze(), self.clean_files[clean_idx], self.noisy_files[idx]


# Lightning DataModule
class AudioDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16, fraction=1.0, authentic=False):
        super(AudioDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers if torch.cuda.is_available() else 1
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
    

def custom_collate_fn(batch):
    # 'batch' should be a list of tuples where each tuple contains data from paired and authentic loaders
    paired_data = [item[0] for item in batch]
    authentic_data = [item[1] for item in batch]
    
    # Process the paired and authentic data
    # Each item in paired_data and authentic_data is a tuple of (clean, noisy)
    clean_paired = torch.stack([b[0] for b in paired_data])
    noisy_paired = torch.stack([b[1] for b in paired_data])
    clean_authentic = torch.stack([b[0] for b in authentic_data])
    noisy_authentic = torch.stack([b[1] for b in authentic_data])
    return (clean_paired, noisy_paired), (clean_authentic, noisy_authentic)


class CombinedDataset(Dataset):
    def __init__(self, paired_dataset, authentic_dataset):
        self.paired_dataset = paired_dataset
        self.authentic_dataset = authentic_dataset

    def __len__(self):
        return min(len(self.paired_dataset), len(self.authentic_dataset))

    def __getitem__(self, idx):
        return self.paired_dataset[idx], self.authentic_dataset[idx]


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
        self.train_dataset_paired = AudioDataset(self.clean_path, self.noisy_path_paired, is_train=True, fraction=self.fraction, authentic=False)
        self.train_dataset_authentic = AudioDataset(self.clean_path, self.noisy_path_authentic, is_train=True, fraction=self.fraction, authentic=True)
        self.val_dataset_paired = AudioDataset(self.test_clean_path, self.test_noisy_path_paired, is_train=False, authentic=False)
        self.val_dataset_authentic = AudioDataset(self.test_clean_path, self.test_noisy_path_authentic, is_train=False, authentic=True)
        # Instantiate the combined dataset
        self.combined_val_dataset = CombinedDataset(self.val_dataset_paired, self.val_dataset_authentic)
        self.combined_train_dataset = CombinedDataset(self.train_dataset_paired, self.train_dataset_authentic)

    def train_dataloader(self):
        return DataLoader(self.combined_train_dataset, batch_size=int(self.batch_size/2), shuffle=True, collate_fn=custom_collate_fn, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.combined_val_dataset, batch_size=int(self.batch_size/2), shuffle=False, collate_fn=custom_collate_fn, num_workers=self.num_workers)


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
        self.dummy_val = DummyDataset(mean_dif=self.mean_dif, mode='val')
        self.save_hyperparameters()

    def setup(self, stage=None):
        self.dummy_train = DummyDataset(mean_dif=self.mean_dif, mode='train')

    def train_dataloader(self):
        return DataLoader(self.dummy_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.dummy_val, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, drop_last=True)
    
class speaker_split_dataset(Dataset):
    def __init__(self, num_speakers, clean_path, noisy_path, fraction=1.0, is_train=True):
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.fraction = fraction
        self.num_speakers = num_speakers

        if is_train:
            self.speakers = set()
            self.clean_files = []
            self.noisy_files = []
            for file in os.listdir(clean_path):
                self.speakers.add(file.split('_')[0])
            self.speakers = list(self.speakers)
            self.speakers = sorted(self.speakers)
            self.speakers = self.speakers[:num_speakers]
            print("Number of speakers:", len(self.speakers))
            print("Speakers:", self.speakers)

            for speaker in self.speakers:
                for file in os.listdir(clean_path):
                    if file.startswith(speaker):
                        self.clean_files.append(file)
                for file in os.listdir(noisy_path):
                    if file.startswith(speaker):
                        self.noisy_files.append(file)

        else:
            self.clean_files = sorted([file for file in os.listdir(clean_path) if file.endswith('.wav')])
            self.noisy_files = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

        num_files = int(fraction * len(self.clean_files))
        self.clean_files = sorted(self.clean_files[:num_files])
        self.noisy_files = sorted(self.noisy_files[:num_files])

    def __len__(self):
        return min(len(self.clean_files), len(self.noisy_files))
    
    def __getitem__(self, idx):
        # Get length of the audio files
        clean_num_frames = torchaudio.info(os.path.join(self.clean_path, self.clean_files[idx])).num_frames
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
        start_frame = random.randint(0, clean_num_frames-2*clean_sample_rate)
        clean_waveform, _ = torchaudio.load(os.path.join(self.clean_path, self.clean_files[idx]), frame_offset=start_frame, num_frames=2*clean_sample_rate, backend='soundfile')
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

        return clean_stft.squeeze(), noisy_stft.squeeze()
    
class SpeakerDataModule(L.LightningDataModule):
    def __init__(self, clean_path, noisy_path, test_clean_path, test_noisy_path, batch_size=16, num_workers=16, fraction=1.0, num_speakers=10):
        super(SpeakerDataModule, self).__init__()
        self.clean_path = clean_path
        self.noisy_path = noisy_path
        self.test_clean_path = test_clean_path
        self.test_noisy_path = test_noisy_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fraction = fraction
        self.num_speakers = num_speakers
        self.save_hyperparameters()

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = speaker_split_dataset(self.num_speakers, self.clean_path, self.noisy_path, fraction=self.fraction, is_train=True)
            self.val_dataset = speaker_split_dataset(self.num_speakers, self.test_clean_path, self.test_noisy_path, fraction=self.fraction, is_train=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers, persistent_workers=True, pin_memory=True, drop_last=True)

if __name__ == '__main__':
    VCTK_clean_path = os.path.join(os.getcwd(), 'data/clean_raw/')
    VCTK_noisy_path = os.path.join(os.getcwd(), 'data/noisy_raw/')

    # Files are names as 'p225_001.wav' where 'p225' is the speaker ID and '001' is the utterance ID
    # Find how many speakers are there in the dataset
    
    num_speakers = 1
    dataset = speaker_split_dataset(num_speakers, VCTK_clean_path, VCTK_noisy_path, fraction=1.)
    print(len(dataset))
    print(dataset[0][0].shape, dataset[0][1].shape)

    data_module = SpeakerDataModule(VCTK_clean_path, VCTK_noisy_path, VCTK_clean_path, VCTK_noisy_path, batch_size=16, num_workers=8, fraction=1.0, num_speakers=num_speakers)
    data_module.setup()
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    for batch in train_loader:
        print(batch[0].shape, batch[1].shape)
        break
    for batch in val_loader:
        print(batch[0].shape, batch[1].shape)
        break
    