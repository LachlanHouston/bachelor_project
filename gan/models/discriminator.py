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
    
def get_gradient_penalty(real_output, fake_output, Discriminator):
    alpha = torch.rand(real_output.shape[0], 1, 1, 1).requires_grad_(True)

    Discriminator = Discriminator(input_sizes=[2, 8, 16, 32, 64, 128], output_sizes=[8, 16, 32, 64, 128, 128])

    difference = fake_output - real_output
    interpolates = real_output + (alpha * difference)
    out = Discriminator(interpolates)

    gradients = torch.autograd.grad(outputs=out, inputs=interpolates, grad_outputs=torch.ones(out.size()), create_graph=True, retain_graph=True, only_inputs=True)[0]
    slopes = torch.sqrt(torch.sum(torch.square(gradients), axis=[1, 2, 3]))
    gradient_penalty = torch.mean((slopes - 1.) ** 2)

    return gradient_penalty

def get_discriminator_loss(real_output, fake_output, alpha=10.):
    real_loss = torch.mean(real_output)
    fake_loss = torch.mean(fake_output)
    gradient_penalty = get_gradient_penalty(real_output, fake_output, Discriminator)
    total_loss = fake_loss - real_loss + alpha * gradient_penalty
    return total_loss

if __name__ == '__main__':
    real1, sample_rate1 = torchaudio.load('data/clean_raw/p226_004.wav')
    real2, _ = torchaudio.load('data/clean_raw/p226_006.wav')
    real3, _ = torchaudio.load('data/clean_raw/p226_009.wav')
    real4, _ = torchaudio.load('data/clean_raw/p226_011.wav')
    real5, _ = torchaudio.load('data/clean_raw/p226_012.wav')
    real6, _ = torchaudio.load('data/clean_raw/p226_015.wav')

    fake1, sample_rate1 = torchaudio.load('data/noisy_raw/p226_005.wav')
    fake2, _ = torchaudio.load('data/noisy_raw/p226_006.wav')
    fake3, _ = torchaudio.load('data/noisy_raw/p226_009.wav')
    fake4, _ = torchaudio.load('data/noisy_raw/p226_011.wav')
    fake5, _ = torchaudio.load('data/noisy_raw/p226_012.wav')
    fake6, _ = torchaudio.load('data/noisy_raw/p226_014.wav')
    sample_rate = sample_rate1
    
    # Cut into 2 second chunks
    waveform1 = real1[:, 0:2*sample_rate]
    waveform2 = real2[:, 0:2*sample_rate]
    waveform3 = real3[:, 0:2*sample_rate]
    waveform4 = real4[:, 0:2*sample_rate]
    waveform5 = real5[:, 0:2*sample_rate]
    waveform6 = real6[:, 0:2*sample_rate]
    real_input = torch.cat((waveform1, waveform2, waveform3, waveform4, waveform5, waveform6), dim=0)

    waveform1 = fake1[:, 0:2*sample_rate]
    waveform2 = fake2[:, 0:2*sample_rate]
    waveform3 = fake3[:, 0:2*sample_rate]
    waveform4 = fake4[:, 0:2*sample_rate]
    waveform5 = fake5[:, 0:2*sample_rate]
    waveform6 = fake6[:, 0:2*sample_rate]
    fake_input = torch.cat((waveform1, waveform2, waveform3, waveform4, waveform5, waveform6), dim=0)

    real_input = torchaudio.transforms.Resample(sample_rate, 16000)(real_input)
    fake_input = torchaudio.transforms.Resample(sample_rate, 16000)(fake_input)

    # Apply STFT
    Xstft_real = torch.stft(real_input, n_fft=512, hop_length=100, win_length=400, return_complex=True, window=torch.hann_window(400))
    Xstft_fake = torch.stft(fake_input, n_fft=512, hop_length=100, win_length=400, return_complex=True, window=torch.hann_window(400))

    x_real = torch.stack((Xstft_real.real, Xstft_real.imag), dim=1)
    x_fake = torch.stack((Xstft_fake.real, Xstft_fake.imag), dim=1)
    print(x_real.shape)
    print(x_fake.shape)

    loss = get_gradient_penalty(x_real, x_fake, Discriminator)
    print(loss)
        
        
