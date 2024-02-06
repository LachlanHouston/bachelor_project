import torch
from gan import Generator
from gan import Discriminator
from gan import Autoencoder
import torchaudio

def predict(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader
) -> None:
    """Run prediction for a given model and dataloader.
    
    Args:
        model: model to use for prediction
        dataloader: dataloader with batches
    
    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)

if __name__ == '__main__':
    model = Autoencoder()

    checkpoint = torch.load('models/epoch=14-val_SNR=-32.94.ckpt')
    model.load_state_dict(checkpoint['state_dict'])

    # Get the generator
    generator = model.generator

    # load a waveform
    waveform, sample_rate = torchaudio.load('data/noisy_processed/p230_080_0.wav')

    # downsample to 16 kHz
    waveform = torchaudio.transforms.Resample(sample_rate, 16000)(waveform)

    # transform the waveform to the STFT
    input = torch.stft(waveform, n_fft=512, hop_length=100, win_length=400, return_complex=True, window=torch.hann_window(400))
    input = torch.stack([input.real, input.imag], dim=1)
    print("input shape:", input.shape)

    # get the output from the generator
    output = generator(input)
    print("output shape:", output.shape)

    # Separate the real and imaginary components
    stft_real = output[:, 0, :, :]
    stft_imag = output[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)

    # convert the output to the waveform
    output = torch.istft(stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400)).detach().cpu().numpy()

    # convert the output to a tensor
    output = torch.tensor(output)
    
    print("output shape:", output.shape)

    # Calculate the norm distance between the input and the output
    norm = torch.norm(waveform - output, p=1)
    print("Norm distance:", norm)

    # save the output to a file
    torchaudio.save('reports/waveform_output.wav', output, 16000)
