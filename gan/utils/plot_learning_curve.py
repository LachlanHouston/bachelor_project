import matplotlib.pyplot as plt
from gan.models.autoencoder import Autoencoder
from gan.models.discriminator import Discriminator
from gan.models.generator import Generator
import os
import torch
import torchaudio
import numpy as np
import csv
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
from gan.utils.utils import stft_to_waveform
from tqdm import tqdm

def compute_scores(real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform):
    ## SI-SNR
    sisnr = ScaleInvariantSignalNoiseRatio().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    sisnr_score = sisnr(preds=fake_clean_waveform, target=real_clean_waveform).item()

    reference_waveforms = non_matching_reference_waveform
    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos_squim_score = torch.mean(subjective_model(fake_clean_waveform.cpu(), reference_waveforms.cpu())).item()

    return sisnr_score, mos_squim_score

def generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=True):
    all_rows = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = generator.to(device)

    for i in tqdm(range(len(clean_files))):
        clean_wav, sr = torchaudio.load(os.path.join(test_clean_path, clean_files[i]))
        noisy_wav, _ = torchaudio.load(os.path.join(test_noisy_path, noisy_files[i]))

        # Choose a randon clean audio file that is not the same as the clean audio file
        random_clean_file = clean_files[i]
        while random_clean_file == clean_files[i]:
            random_clean_file = np.random.choice(clean_files)

        random_clean_wav, _ = torchaudio.load(os.path.join(test_clean_path, random_clean_file))

        # Resample to 16kHz
        clean_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(clean_wav).to(device)
        noisy_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(noisy_wav).to(device)
        random_clean_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(random_clean_wav).to(device)

        # Compute the STFT of the audio
        clean_stft = torch.stft(clean_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)
        noisy_stft = torch.stft(noisy_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)

        # Stack the real and imaginary parts of the STFT
        clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
        noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)

        fake_clean_stft, _ = generator(noisy_stft)

        real_clean_waveform = stft_to_waveform(clean_stft, device=device).detach()
        fake_clean_waveform = stft_to_waveform(fake_clean_stft, device=device).detach()

        sisnr_score, mos_score = compute_scores(real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform = random_clean_wav)

        all_rows.append([sisnr_score, mos_score])
        
    ## Means
    print(f"Mean: {np.mean(all_rows, axis=0)}")

    ## Standard errors of the means
    print(f"SE of the means: {np.std(all_rows, axis=0) / np.sqrt(len(all_rows))}")

if __name__ == '__main__':
    model_names = ['10p.ckpt', '20p.ckpt', '30p.ckpt', '40p.ckpt', '50p.ckpt', '60p.ckpt', '70p.ckpt', '80p.ckpt', '90p.ckpt']

    clean_files = sorted(os.listdir(os.path.join(os.getcwd(), 'data/clean_raw/')))
    noisy_files = sorted(os.listdir(os.path.join(os.getcwd(), 'data/noisy_raw/')))
    test_clean_files = sorted(os.listdir(os.path.join(os.getcwd(), 'data/test_clean_raw/')))
    test_noisy_files = sorted(os.listdir(os.path.join(os.getcwd(), 'data/test_noisy_raw/')))

    train_snr = []
    test_snr = []

    # Loop through all models
    for model_name in model_names:
        cpkt_path = os.path.join('models', model_name)
        cpkt_path = os.path.join(os.getcwd(), cpkt_path)
        model = Autoencoder.load_from_checkpoint(cpkt_path).generator
        
        #SNR_train = generator_scores(model, os.path.join(os.getcwd(), 'data/clean_raw/'), os.path.join(os.getcwd(), 'data/noisy_raw/'), clean_files, noisy_files, model_name + "_train", use_pesq=True)
        SNR_test = generator_scores(model, os.path.join(os.getcwd(), 'data/test_clean_raw/'), os.path.join(os.getcwd(), 'data/test_noisy_raw/'), test_clean_files, test_noisy_files, model_name + "_test", use_pesq=True)





