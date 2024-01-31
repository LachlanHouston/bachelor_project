import torch.nn as nn
import torch
import torchaudio

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)):
        super().__init__()
        norm_f = nn.utils.spectral_norm
        self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        self.activation = nn.LeakyReLU(0.1)
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_sizes=[], output_sizes=[]):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes

        norm_f = nn.utils.spectral_norm

        assert len(self.input_sizes) == len(self.output_sizes), "Input and output sizes must be the same length"

        for i in range(len(self.input_sizes)):
            self.conv_layers.append(Conv2DBlock(self.input_sizes[i], self.output_sizes[i], kernel_size=(5, 5), stride=(2, 2)))
            
        self.fc_layers1  = norm_f(nn.Linear(256, 64))
        self.activation = nn.LeakyReLU(0.1)
        self.fc_layers2  = norm_f(nn.Linear(64, 1))

    def forward(self, x) -> torch.Tensor:
        for i in range(len(self.input_sizes)):
            x = self.conv_layers[i](x)
        x = x.flatten(1, -1)
        x = self.fc_layers1(x)
        x = self.activation(x)
        x = self.fc_layers2(x)
        
        return x
    

if __name__ == '__main__':
    waveform1, sample_rate1 = torchaudio.load('data/clean_raw/p226_006.wav')
    #waveform2, sample_rate2 = torchaudio.load('data/clean_raw/p226_009.wav')
    sample_rate = sample_rate1
    
    # Cut into 2 second chunks
    waveform1 = waveform1[:, :sample_rate*2]
    #waveform2 = waveform2[:, :sample_rate*2]

    #waveform = torch.cat((waveform1, waveform2), dim=0)

    # Downsample to 16 kHz
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform1)
    sample_rate = 16000

    # Apply STFT
    Xstft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True, window=torch.hann_window(400))
    x = torch.stack((Xstft.real, Xstft.imag), dim=1)
    print(x.shape)

    model = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])
    out = model(x)
    print(out.shape)
    print(out)
        
        
