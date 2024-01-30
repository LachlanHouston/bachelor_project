import torch.nn as nn
import torch.nn.functional as F
import torch

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 2), 
                 padding=(0, 1), d_norm='spectral_norm'):
        super(Conv2DBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)



    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, num_conv_layers=6, num_fc_layers=2):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.fc_layers = nn.ModuleList()
        self.num_conv_layers = num_conv_layers
        self.num_fc_layers = num_fc_layers
        self.conv_layers.append(Conv2DBlock(2, 8, kernel_size=(5, 5), stride=(2, 2)))
        self.conv_layers.append(Conv2DBlock(8, 16, kernel_size=(5, 5), stride=(2, 2)))
        self.conv_layers.append(Conv2DBlock(16, 32, kernel_size=(5, 5), stride=(2, 2)))
        self.conv_layers.append(Conv2DBlock(32, 64, kernel_size=(5, 5), stride=(2, 2)))
        self.conv_layers.append(Conv2DBlock(64, 128, kernel_size=(5, 5), stride=(2, 2)))
        self.conv_layers.append(Conv2DBlock(128, 128, kernel_size=(5, 5), stride=(2, 2)))
        self.fc_layers.append(nn.Linear(256, 64))
        self.fc_layers.append(nn.Linear(64, 1))

    def forward(self, x):
        for i in range(self.num_conv_layers):
            x = self.conv_layers[i](x)
            print(i, x.shape)
        x = torch.flatten(x, 1, -1)
        print(x.shape)
        for i in range(self.num_fc_layers):
            x = self.fc_layers[i](x)
            print(x.shape)
        return x
    

if __name__ == '__main__':
    import torchaudio
    waveform, sample_rate = torchaudio.load('data/clean_raw/p226_004.wav', normalize=True)
    print(waveform.shape)

    # Downsample to 16 kHz
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    sample_rate = 16000

    if len(waveform.shape) == 3 and waveform.shape[1] == 1:
        waveform = waveform.squeeze(1)
        print("yes")
    
    # Apply STFT
    Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True)
    data_list = [Xstft.real, Xstft.imag]
    data = torch.cat(data_list, dim=1)
    print(data.shape)

    x = data


    model = Discriminator()
    out = model(x)
    print(out.shape)

    # Load file from data/clean_raw/p226_004.wav
    # import torchaudio
    # waveform, sample_rate = torchaudio.load('data/clean_raw/p226_004.wav', normalize=True)
    # print(waveform.shape)

    # # Downsample to 16 kHz
    # waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)
    # sample_rate = 16000

    # # Split into 2 second chunks
    # for i in range(0, waveform.shape[1], 32000):
    #     chunk = waveform[:, i:i+32000]
    #     print(chunk.shape)


    # # Apply STFT
    # # spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=100, win_length=400)(waveform)

    # # Another way
    # Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True)
    # print(Xstft.shape)
    # data_list = [Xstft.real, Xstft.imag]
    # data = torch.cat(data_list, dim=1)
    # print(data.shape)
        
        
