import torchaudio
import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, path, new_sample_rate=16000, n_seconds=2):
        super(AudioDataset, self).__init__()
        self.path = path
        self.files = os.listdir(path)
        self.new_sample_rate = new_sample_rate
        self.n_seconds = n_seconds

    def __len__(self):
        return len(self.files)
    
    def transform(self, waveform, sample_rate):
        """Transform the waveform"""
        # Downsample to 16 kHz
        waveform = torchaudio.transforms.Resample(sample_rate, self.new_sample_rate)(waveform)
        
        # Cut into n second chunks
        chunks = []
        for i in range(0, waveform.shape[1], self.new_sample_rate * self.n_seconds):
            chunk = waveform[:, i:i+self.new_sample_rate*self.n_seconds]
            if chunk.shape[1] != self.new_sample_rate * self.n_seconds:
                chunk = F.pad(chunk, (0, self.new_sample_rate * self.n_seconds - chunk.shape[1]))

            chunk_stft = torch.stft(chunk, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
            chunk_stft = torch.stack((chunk_stft.real, chunk_stft.imag), dim=1)
            chunks.append(chunk_stft)
        print(len(chunks))
        return torch.stack(chunks, dim=2)



    def __getitem__(self, idx, transform=True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        print(self.path + self.files[idx])
        
        waveform, sample_rate = torchaudio.load(self.path + self.files[idx])
        stft = None

        if transform:
            stft = self.transform(waveform, sample_rate)

        return waveform, sample_rate, stft
    
def collate_fn(batch):
    """Collate function for the DataLoader"""
    waveforms = []
    sample_rates = []
    stft = []
    for waveform, sample_rate, spectrogram in batch:
        waveforms.append(waveform)
        sample_rates.append(sample_rate)
        stft.append(spectrogram)
    
    return waveforms, sample_rates, stft

if __name__ == '__main__':
    clean_processed_path = 'data/clean_raw/'
    clean_dataset = AudioDataset(clean_processed_path)
    clean_dataloader = DataLoader(clean_dataset, batch_size=2, shuffle=False, collate_fn=collate_fn)
    for waveforms, sample_rates, spectrograms in clean_dataloader:
        print(waveforms[1].shape)
        print(sample_rates[1])
        print(spectrograms[1].shape)
        break