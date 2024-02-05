import torchaudio
import os
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from scipy.io import wavfile

class AudioDataset(Dataset):
    def __init__(self, clean_path, noisy_path,
                 new_sample_rate=16000):
        super(AudioDataset, self).__init__()
        self.clean_path = clean_path
        self.clean_files = [file for file in os.listdir(clean_path) if file.endswith('.wav')]

        self.noisy_path = noisy_path
        self.noisy_files = [file for file in os.listdir(noisy_path) if file.endswith('.wav')]

        self.new_sample_rate = new_sample_rate

    def __len__(self):
        return len(self.clean_files)
    
    def __getitem__(self, idx):
        clean_file = self.clean_files[idx]
        noisy_file = self.noisy_files[idx]
        clean_waveform, _ = torchaudio.load(self.clean_path + clean_file)
        noisy_waveform, _ = torchaudio.load(self.noisy_path + noisy_file)

        # Transform with ftst
        clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

        clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
        noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)

        return clean_stft, noisy_stft
    
def collate_fn(batch):
    clean_waveforms, noisy_waveforms = zip(*batch)
    return clean_waveforms, noisy_waveforms

def stft_to_waveform(stft):
    if len(stft.shape) == 3:
        stft = stft.unsqueeze(0)
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

def data_loader(clean_path, noisy_path, split =[0.8, 0.1, 0.1],
                batch_size=16, num_workers=4):
    
    dataset = AudioDataset(clean_path, noisy_path)
    print('Dataset:', len(dataset))
    train_size = int(split[0] * len(dataset))
    val_size = int(split[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn, drop_last=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, drop_last=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_fn, drop_last=True, persistent_workers=True)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    clean_processed_path = 'data/clean_processed/'
    noisy_processed_path = 'data/noisy_processed/'
    
    train_loader, val_loader, test_loader = data_loader(clean_processed_path, noisy_processed_path)
    print('Train:', len(train_loader), 'Validation:', len(val_loader), 'Test:', len(test_loader))
    for batch in train_loader:
        clean_waveforms, noisy_waveforms = batch
        print(clean_waveforms[0].size(), noisy_waveforms[0].size())
        clean_wav = stft_to_waveform(clean_waveforms[0])
        wavfile.write('clean.wav', 16000, clean_wav.numpy().reshape(-1))
        torch.save(clean_wav, 'clean.wav', format = 'wav')
        print(clean_wav.size())
        break
    for batch in val_loader:
        clean_waveforms, noisy_waveforms = batch
        print(clean_waveforms[0].size(), noisy_waveforms[0].size())
        break
    for batch in test_loader:
        clean_waveforms, noisy_waveforms = batch
        print(clean_waveforms[0].size(), noisy_waveforms[0].size())
        break
    