import sys
import os
base_path = os.path.expanduser('~/bachelor_project')
if base_path not in sys.path:
    sys.path.insert(0, base_path)
    sys.path =  base_path
print("sys.path:", sys.path)
from gan.models.DPRNN import DPRNN
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio

plot = False

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
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

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
            x = x[:, :, :, :-1]  # chomp size
        if self.is_last is False:
            x = self.norm(x)
            x = self.activation(x)
        return x
    
    
class Generator(torch.nn.Module):
    def __init__(self, param, in_channels=2, **kwargs):
        super().__init__()
        self.debug = False
        self.mask = kwargs.get('mask', True)     
        self.mask_bound = kwargs.get('mask_bound', 'tanh')
        param["encoder"][0][0] = in_channels
        param["decoder"][-1][1] = in_channels
        self.encoder = nn.ModuleList([ConvBlock(*item) for item in param["encoder"]])
        self.decoder = nn.ModuleList([TransConvBlock(*item) for item in param["decoder"]])
        
        rnn_kwargs = {            
            "encoder_dim": param["encoder"][-1][1],
            "num_layers": kwargs.get('rnn_layers', 4),
            "bidirectional": kwargs.get('bidirectional', False)
            }
        self.rnn_block = DPRNN(**rnn_kwargs, hidden_size=kwargs.get('hidden_size', 128))
        
    def forward(self, x):
        # x: shape of [batch size, in_channels, n_fft//2+1, T]
        e = x
        e_list = []
        for i, layer in enumerate(self.encoder):
            e = layer(e)
            e_list.append(e)
            if self.debug:
                print(f"encoder_{i}: {e.shape}")
        rnn_out = self.rnn_block(e)
        if self.debug:
             print(f"rnn_out: {rnn_out.shape}")
        idx = len(e_list)
        d = rnn_out        
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            d = layer(_padded_cat(d, e_list[idx]))
            if self.debug:
                print(f"decoder_{i}: {d.shape}")
        if self.mask is True:
            d = getattr(torch, self.mask_bound)(d)
        return d
    

# Example usage

# Assuming the DPRNN class and other dependencies are defined as per your code
# Define parameters for the generator
    # generator
# model_nlayer_nchannel
params = {
    # (in_channels, out_channels, kernel_size, stride, padding)
    "encoder":
        [[  0,  32, (5, 2), (2, 1), (1, 1)],
         ( 32,  64, (5, 2), (2, 1), (2, 1)),
         ( 64, 128, (5, 2), (2, 1), (2, 1))],
    # (in_channels, out_channels, kernel_size, stride, padding, output_padding, is_last)
    "decoder":
        [(256,  64, (5, 2), (2, 1), (2, 0), (1, 0)),
         (128,  32, (5, 2), (2, 1), (2, 0), (1, 0)),
         [ 64,   0, (5, 2), (2, 1), (1, 0), (0, 0), True]]
}

if __name__ == '__main__':
    # Load the waveform
    in_waveform, sample_rate = torchaudio.load('data/clean_raw/p226_004.wav', normalize=True)

    # Downsample to 16 kHz
    in_waveform = torchaudio.transforms.Resample(sample_rate, 16000)(in_waveform)
    sample_rate = 16000

    # Apply STFT
    Xstft = torch.stft(in_waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True)
    input = torch.stack([Xstft.real, Xstft.imag], dim=1)
    # input = torch.cat(input_list, dim=1)
    print("input shape:", input.shape)

    # Initialize the generator
    generator = Generator(param=params, in_channels=2, debug=True)

    # Get the output from the generator
    output = generator(input)
    print("Output shape:", output.shape)

    '''Convert to waveform'''
    # Separate the real and imaginary components
    Ystft_real = output[:, 0, :, :]
    Ystft_imag = output[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    Ystft = torch.complex(Ystft_real, Ystft_imag)
    # Perform inverse STFT to obtain the waveform
    out_waveform = torch.istft(Ystft, n_fft=512, hop_length=100, win_length=400)

    # Print the shape of the waveform tensor
    print("Shape of in_waveform:", in_waveform.shape)
    print("Shape of out_waveform:", out_waveform.shape)

    if plot:
        '''Plot old waveform'''
        # Convert the waveform tensor to a numpy array
        waveform_np = in_waveform.squeeze().detach().numpy()
        # Create a time axis for the waveform
        time_axis = torch.arange(0, waveform_np.shape[0])
        # Plot the waveform
        plt.plot(time_axis, waveform_np)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('Old Waveform')
        plt.show()

        '''Plot new waveform'''
        # Convert the waveform tensor to a numpy array
        waveform_np = out_waveform.squeeze().detach().numpy()
        # Create a time axis for the waveform
        time_axis = torch.arange(0, waveform_np.shape[0])
        # Plot the waveform
        plt.plot(time_axis, waveform_np)
        plt.xlabel('Time')
        plt.ylabel('Amplitude')
        plt.title('New Waveform')
        plt.show()





