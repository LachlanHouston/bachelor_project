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
        return len(self.clean_files)
    
    def transform(self, waveform, sample_rate, max_length):
        # Downsample to 16 kHz
        waveform = torchaudio.transforms.Resample(sample_rate, self.new_sample_rate)(waveform)
        
        # Pad the waveform to have the same length
        waveform = F.pad(waveform, (0, max_length - waveform.shape[1]))

        # Process with stft
        Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hamming_window(400), return_complex=True)
        data_list = [Xstft.real, Xstft.imag]
        data = torch.cat(data_list, dim=1)

        return data

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

if __name__ == '__main__':
    clean_processed_path = 'data/clean_raw/'
    noisy_processed_path = 'data/noisy_raw/'
    
    dataset = AudioDataset(clean_processed_path, noisy_processed_path)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_fn)

    for i, (clean_stft, noisy_stft) in enumerate(dataloader):
        print(clean_stft[0].shape)
        print(noisy_stft[0].shape)
        break