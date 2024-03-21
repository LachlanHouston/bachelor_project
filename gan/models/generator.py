from gan.models.DPRNN import DPRNN
import torch
from torch import nn
import torch.nn.functional as F

def _padded_cat(x, y, dim=1):
    # Pad x to have same size with y, and cat them
    # x dim: N, C, T, F
    x_pad = F.pad(x, (0, y.shape[3] - x.shape[3], 
                      0, y.shape[2] - x.shape[2])) # pad T, F
    z = torch.cat((x_pad, y), dim=dim) # cat on C
    return z

class ConvBlock(nn.Module):
    "norm: weight, batch, layer, instance"
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1), 
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
    def __init__(self, in_channels, out_channels, kernel_size=(5, 2), stride=(2, 1),
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
    def __init__(self, param=None, in_channels=2, out_channels=2):
        super().__init__()
        self.encoder = nn.ModuleList([])
        self.decoder = nn.ModuleList([])
        self.rnn_block = DPRNN(128, rnn_type='LSTM', hidden_size=128, output_size=128, num_layers=2, bidirectional=True)
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Encoder
        self.encoder.append(ConvBlock(self.in_channels, 32, kernel_size=(5, 2), stride=(2, 1), padding=(1, 1))) # B, 32, 129, 321
        self.encoder.append(ConvBlock(32, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1))) # B, 64, 65, 321
        self.encoder.append(ConvBlock(64, 128, kernel_size=(5, 2), stride=(2, 1), padding=(2, 1))) # B, 128, 33, 321

        # Decoder
        self.decoder.append(TransConvBlock(256, 64, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))) # B, 64, 65, 321
        self.decoder.append(TransConvBlock(128, 32, kernel_size=(5, 2), stride=(2, 1), padding=(2, 0), output_padding=(1, 0))) # B, 32, 129, 321
        self.decoder.append(TransConvBlock(64, self.out_channels, kernel_size=(5, 2), stride=(2, 1), padding=(1, 0), output_padding=(0, 0), is_last=True)) # B, 2, 257, 321

        self.activation = nn.Tanh()

    def forward(self, x):
        e = x[:, :self.in_channels, :, :] # Include phase or only magnitude
        e_list = []
        """Encoder"""
        for i, layer in enumerate(self.encoder):
            # apply convolutional layer
            e = layer(e)
            # store the output for skip connection
            e_list.append(e)
        
        """Dual-Path RNN"""
        rnn_out = self.rnn_block(e) # [32, 128, 32, 321]
        # store length to go through the list backwards
        idx = len(e_list)
        d = rnn_out

        """Decoder"""
        for i, layer in enumerate(self.decoder):
            idx = idx - 1
            # concatenate d with the skip connection and put though layer
            d = layer(_padded_cat(d, e_list[idx]))

        d = self.activation(d)
        mask = d
        if mask.shape[1] != x.shape[1]:
            # Add mask to first channel of x (concat 0 to the channel dimension)
            mask = torch.cat((mask, torch.zeros((mask.shape[0], 1, mask.shape[2], mask.shape[3]), device=mask.device)), dim=1)

        # Perform hadamard product
        output = torch.mul(x, mask)
        
        return output, mask

if __name__ == '__main__':
    input = torch.normal(0, 1, (4, 2, 257, 321))

    # Initialize the generator
    generator = Generator(in_channels=2, out_channels=2)

    # Get the output from the generator
    output, mask = generator(input)
    print("Output shape:", output.shape)
    print("Mask shape:", mask.shape)





