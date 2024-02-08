from gan.models.DPRNN import DPRNN
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
import matplotlib.pyplot as plt

def _padded_cat(x, y, dim=1):
    # Pad x to have same size with y, and cat them
    x_pad = F.pad(x, (0, y.shape[3] - x.shape[3], 
                      0, y.shape[2] - x.shape[2]))
    z = torch.cat((x_pad, y), dim=dim)
    return z

class ConvBlock(nn.Module):
    "norm: weight, batch, layer, instance"
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), 
                 padding=(0, 1), causal=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)        
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.PReLU()
        self.causal = causal        

    def forward(self, x):
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]  # chomp size
        x = self.norm(x)
        x = self.activation(x)
        return x

class TransConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1),
                 padding=(0, 0), output_padding=(0, 0), is_last=False, causal=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, 
                                       padding, output_padding)        
        self.norm = nn.BatchNorm2d(out_channels)
        self.is_last = is_last
        self.causal = causal
        self.activation = nn.PReLU()
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        x = self.conv(x)
        if self.causal is True:
            x = x[:, :, :, :-1]
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x
    
    
class Generator(nn.Module):
    def __init__(self, param=None, in_channels=2):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.rnn_block = DPRNN(128)
        self.in_channels = in_channels
        self.out_channels = 2

        # Encoder
        self.encoder.append(ConvBlock(in_channels, 32, kernel_size=(5, 2), stride=(2, 1), padding=(1, 1)))
        self.encoder.append(ConvBlock(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))
        self.encoder.append(ConvBlock(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1)))

        # Decoder
        self.decoder.append(TransConvBlock(256, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)))
        self.decoder.append(TransConvBlock(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0)))
        self.decoder.append(TransConvBlock(64, 2, kernel_size=(5, 2), stride=(2, 1), padding=(1, 0), output_padding=(0, 0), is_last=True))

        self.activation = nn.Tanh()

    def forward(self, x):
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
        rnn_out = self.rnn_block(e) # [32, 128, 32, 321]
        idx = len(e_list)
        d = rnn_out
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            d = layer(_padded_cat(d, e_list[idx]))
        d = self.activation(d)
        return d

# def stft_to_waveform(stft):
#     # Separate the real and imaginary components
#     stft_real = stft[:, 0, :, :]
#     stft_imag = stft[:, 1, :, :]
#     # Combine the real and imaginary components to form the complex-valued spectrogram
#     stft = torch.complex(stft_real, stft_imag)
#     # Perform inverse STFT to obtain the waveform
#     waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400)
#     return waveform

def waveform_to_stft(waveform):
    # Perform STFT to obtain the complex-valued spectrogram
    stft = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True, window=torch.hann_window(400))
    # Separate the real and imaginary components
    stft = torch.stack([stft.real, stft.imag], dim=1)
    return stft

if __name__ == '__main__':
    # Load the waveform
    in_waveform, sample_rate = torchaudio.load('data/clean_processed/p230_074_0.wav')

    # Downsample to 16 kHz
    in_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(in_waveform)
    sample_rate = 16000
    
    input = waveform_to_stft(in_waveform)

    print("input shape:", input.shape)

    input = torch.rand(16, 2, 257, 321)

    # Initialize the generator
    generator = Generator()

    # Get the output from the generator
    output = generator(input)
    print("Output shape:", output.shape)

    # out_waveform = stft_to_waveform(output)

    # # Print the shape of the waveform tensor
    # print("Shape of in_waveform:", in_waveform.shape)
    # print("Shape of out_waveform:", out_waveform.shape)

    # if plot:
    #     '''Plot old waveform'''
    #     # Convert the waveform tensor to a numpy array
    #     waveform_np = in_waveform.squeeze().detach().numpy()
    #     # Create a time axis for the waveform
    #     time_axis = torch.arange(0, waveform_np.shape[0])
    #     # Plot the waveform
    #     plt.plot(time_axis, waveform_np)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title('Old Waveform')
    #     plt.show()

    #     '''Plot new waveform'''
    #     # Convert the waveform tensor to a numpy array
    #     waveform_np = out_waveform.squeeze().detach().numpy()
    #     # Create a time axis for the waveform
    #     time_axis = torch.arange(0, waveform_np.shape[0])
    #     # Plot the waveform
    #     plt.plot(time_axis, waveform_np)
    #     plt.xlabel('Time')
    #     plt.ylabel('Amplitude')
    #     plt.title('New Waveform')
    #     plt.show()






