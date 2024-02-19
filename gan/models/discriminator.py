import torch.nn as nn
import torch

class Conv2DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 2), stride=(2, 1), padding=(0, 0)):
        super().__init__()
        # norm_f = nn.utils.spectral_norm
        # self.conv = norm_f(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding))
        # above line is replaced by:
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.activation = nn.LeakyReLU(0.1)
        
        nn.init.xavier_uniform_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)

    def forward(self, x) -> torch.Tensor:
        x = self.conv(x)
        x = self.activation(x)
        return x
    
class Discriminator(nn.Module):
    def __init__(self, input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128]):
        super(Discriminator, self).__init__()
        self.conv_layers = nn.ModuleList()
        self.input_sizes = input_sizes
        self.output_sizes = output_sizes
        # norm_f = nn.utils.spectral_norm
        

        assert len(self.input_sizes) == len(self.output_sizes), "Input and output sizes must be the same length"

        for i in range(len(self.input_sizes)):
            self.conv_layers.append(Conv2DBlock(self.input_sizes[i], self.output_sizes[i], kernel_size=(5, 5), stride=(2, 2)))
            
        # self.fc_layers1  = norm_f(nn.Linear(256, 64))
        # self.activation = nn.LeakyReLU(0.1)
        # self.fc_layers2 = norm_f(nn.Linear(64, 1))
        
        # above three lines are replaced by:
        self.fc_layers1  = nn.Linear(256, 64)
        self.activation = nn.LeakyReLU(0.1)
        self.fc_layers2 = nn.Linear(64, 1)

    def forward(self, x) -> torch.Tensor:
        for layer in self.conv_layers:
            x = layer(x)
        x = x.flatten(1, -1)
        x = self.fc_layers1(x)
        x = self.activation(x)
        x = self.fc_layers2(x)
        
        return x


if __name__ == '__main__':

    Xstft_real = torch.randn(16, 2, 257, 321)
    Xstft_fake = torch.randn(16, 2, 257, 321)

    model = Discriminator()

    output_real = model(Xstft_real)
    output_fake = model(Xstft_fake)

    print(output_real.shape)
    print(output_fake.shape)
        
        
