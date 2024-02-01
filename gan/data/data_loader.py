import torchaudio
import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path, new_sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = [file for file in os.listdir(clean_path) if file.endswith('.wav')]

        self.noisy_path = noisy_path
        self.noisy_files = [file for file in os.listdir(noisy_path) if file.endswith('.wav')]

        self.new_sample_rate = new_sample_rate

    def __len__(self):
        return len(self.noisy_files)
    
    def transform(self, waveform, sample_rate, max_length):
        # Cut the waveform to 2 seconds
        waveform = waveform[:, 0:2*sample_rate]

        # If the waveform is shorter than 2 seconds, pad it with zeros
        if waveform.shape[1] < 2*sample_rate:
            waveform = F.pad(waveform, (0, 2*sample_rate - waveform.shape[1]), 'constant', 0)

        # Downsample to 16 kHz
        waveform = torchaudio.transforms.Resample(sample_rate, self.new_sample_rate)(waveform)

        # Process with stft
        Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hamming_window(400), return_complex=True)
        stft = torch.stack([Xstft.real, Xstft.imag], dim=1)
        return stft

    def __getitem__(self, idx, transform=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        clean_file = self.clean_files[idx]
        noisy_file = self.noisy_files[idx]

        clean_waveform, clean_sample_rate = torchaudio.load(self.clean_path + clean_file)
        noisy_waveform, noisy_sample_rate = torchaudio.load(self.noisy_path + noisy_file)

        clean_stft = None
        noisy_stft = None
        max_length = 725379

        if transform:
            clean_stft = self.transform(clean_waveform, clean_sample_rate, max_length)
            noisy_stft = self.transform(noisy_waveform, noisy_sample_rate, max_length)

        return clean_stft, noisy_stft
    
def collate_fn(batch):
    clean_stft, noisy_stft = zip(*batch)
    return clean_stft, noisy_stft

def stft_to_waveform(stft):
    # Separate the real and imaginary components
    stft_real = stft[:, 0, :, :]
    stft_imag = stft[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)
    # Perform inverse STFT to obtain the waveform
    waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400)
    return waveform

def waveform_to_stft(waveform):
    # Perform STFT to obtain the complex-valued spectrogram
    stft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True)
    # Separate the real and imaginary components
    stft = torch.stack([stft.real, stft.imag], dim=1)
    return stft

def data_loader(clean_path, noisy_path, batch_size=32, num_workers=4):
    dataset = AudioDataset(clean_path, noisy_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=num_workers, drop_last=True)
    return loader


if __name__ == '__main__':
    clean_processed_path = 'data/clean_raw/'
    noisy_processed_path = 'data/noisy_raw/'
    
    dataset = AudioDataset(clean_processed_path, noisy_processed_path)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn, drop_last=True)
    
    for i, (clean_stft, noisy_stft) in enumerate(loader):
        print(clean_stft[0].shape)
        print(noisy_stft[0].shape)
        break