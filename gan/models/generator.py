import torch
import torch.nn as nn
import torch.nn.functional as F

input = torch.zeros([1, 2, 257, 317])


class UNetEncoder(nn.Module):
    
    class UNetEncoder(nn.Module):
        def __init__(self, in_channels, out_channels, num_layers):
            super(UNetEncoder, self).__init__()
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.PReLU()
            ]

            for _ in range(num_layers - 1):
                layers += [
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.PReLU()
                ]

            self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)

# Example usage
encoder = UNetEncoder(in_channels=1, out_channels=64, num_layers=2)
print(input.shape)
print(encoder.forward(input).shape)

#%%

class DualPathBlock(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(DualPathBlock, self).__init__()
        self.freq_lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.time_lstm = nn.LSTM(hidden_size, hidden_size, batch_first=True)

    def forward(self, x):
        # x shape: [batch, time, frequency]
        # Process across frequency
        freq_out, _ = self.freq_lstm(x)
        # Reshape for time processing
        time_out = freq_out.transpose(1, 2)  # swap time and frequency dimensions
        # Process across time
        final_out, _ = self.time_lstm(time_out)
        return final_out
    
# Example usage
# block = DualPathBlock(input_size=64, hidden_size=128)
# print(block.forward(input[0]).shape)
