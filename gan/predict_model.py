import torch
from gan import Generator
from gan import Discriminator
from gan import Autoencoder
import torchaudio
from gan import VCTKDataModule
from pytorch_lightning import Trainer
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from speechmos import dnsmos
from gan.utils.utils import SegSNR
import os
import hydra
import numpy as np
from tqdm import tqdm
import csv
import random
torch.set_grad_enabled(False)


def stft_to_waveform(stft, device=torch.device('cuda')):
    if len(stft.shape) == 3:
        stft = stft.unsqueeze(0)
    # Separate the real and imaginary components
    stft_real = stft[:, 0, :, :]
    stft_imag = stft[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)
    # Perform inverse STFT to obtain the waveform
    waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device))
    return waveform

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


@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), 'models/epoch=999.ckpt')
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # Load the trained model
    model.load_state_dict(checkpoint['state_dict'])
    # Get the generator
    generator = model.generator

    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_clean_stft/')
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), 'data/test_noisy_stft/')

    clean_files = [file for file in os.listdir(test_clean_path) if file.endswith('.pt')]
    noisy_files = [file for file in os.listdir(test_noisy_path) if file.endswith('.pt')]


    with open('scores.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SNR", "DNSMOS", "MOS SQUIM"])

        for i in tqdm(range(10)):
            clean_stft = torch.load(os.path.join(test_clean_path, clean_files[i])).requires_grad_(False)
            noisy_stft = torch.load(os.path.join(test_noisy_path, noisy_files[i])).requires_grad_(False)

            fake_clean, mask = generator(noisy_stft)

            real_clean_waveform = stft_to_waveform(clean_stft, device=torch.device('cpu')).detach()
            fake_clean_waveform = stft_to_waveform(fake_clean, device=torch.device('cpu')).detach()

            # torchaudio.save(f'fake_clean_waveform_{i}.wav', fake_clean_waveform, 16000)

            real_clean_waveform = real_clean_waveform.squeeze()
            fake_clean_waveform = fake_clean_waveform.squeeze()

            ## Scale Invariant Signal-to-Noise Ratio
            sisnr = ScaleInvariantSignalNoiseRatio().to(device)
            sisnr_score = sisnr(preds=fake_clean_waveform, target=real_clean_waveform)

            # ## Perceptual Evaluation of Speech Quality
            # pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
            # pesq_score = pesq(real_clean_waveform, fake_clean_waveform)

            ## Deep Noise Suppression Mean Opinion Score (DNSMOS)
            dnsmos_score = dnsmos.run(fake_clean_waveform.numpy(), 16000)['ovrl_mos']


            reference_index = random.choice([j for j in range(len(clean_files)) if j != i])
            reference_waveform = torch.load(os.path.join(test_clean_path, clean_files[reference_index])).requires_grad_(False)
            reference_waveform = stft_to_waveform(reference_waveform.squeeze(0), device=torch.device('cpu')).detach()
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim = subjective_model(fake_clean_waveform.unsqueeze(0), reference_waveform)

            # ## Extended Short Time Objective Intelligibility
            # estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
            # estoi_score = estoi(preds = fake_clean_waveform, target = real_clean_waveform)
            
            # ## Segmental Signal-to-Noise Ratio (SegSNR)
            # segsnr = SegSNR(seg_length=160) # 160 corresponds to 10ms of audio with sr=16000
            # segsnr.update(preds=fake_clean_waveform.unsqueeze(0), target=real_clean_waveform.unsqueeze(0))
            # segsnr_score = segsnr.compute() # Average SegSNR

            # ## Distance between real clean and fake clean
            # dist = torch.norm(real_clean_waveform - fake_clean_waveform, p=1)
            
            writer.writerow([sisnr_score.item(), dnsmos_score, mos_squim.item()])
            







if __name__ == '__main__':
    main()
