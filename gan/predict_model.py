import torch
from gan import Generator
from gan import Discriminator
from gan import Autoencoder
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio

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

    checkpoint = torch.load('models/epoch=999.ckpt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['state_dict'])

    # Get the generator
    generator = model.generator

    # load a waveform
    noisy_stft = torch.load('data/test_noisy_stft/p232_015_2.pt')[0].unsqueeze(0)
    clean_stft = torch.load('data/test_clean_stft/p232_015_2.pt')[0].unsqueeze(0)

    fake_stft, _ = generator(noisy_stft)
    # remove tuple by first turning it into a list and then into a tensor
    # fake_stft = torch.stack([fake_stft[0], fake_stft[1]], dim=0)

    fake_stft = torch.complex(fake_stft[:, 0, :, :], fake_stft[:, 1, :, :])
    clean_stft = torch.complex(clean_stft[:, 0, :, :], clean_stft[:, 1, :, :])

    # Inverse STFT
    fake_waveform = torch.istft(fake_stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))
    clean_waveform = torch.istft(clean_stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400))

    # Calculate the SI-SNR
    sisnr = ScaleInvariantSignalNoiseRatio()
    score = sisnr(fake_waveform, clean_waveform)

    print(f'SI-SNR: {score}')
