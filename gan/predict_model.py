import torch
import torchaudio
from models.generator import Generator
from models.discriminator import Discriminator
from models.autoencoder import Autoencoder
from data.data_loader import AudioDataModule, AudioDataset
from torch.utils.data import DataLoader
from utils.utils import stft_to_waveform, compute_scores
from pytorch_lightning import Trainer
import os
import hydra
import numpy as np
from tqdm import tqdm
import csv
import random
import numpy as np
torch.set_grad_enabled(False)

test_clean_dir = 'data/test_clean_raw/'
test_noisy_dir = 'data/test_noisy_raw/'
model_path = "models/standardmodel.ckpt"
get_generator_scores = True
get_discriminator_scores = False
limit_samples = False     # False: use all samples, Integer: use only the first n samples
use_pesq = True


def load_model(cpkt_path):
    cpkt_path = os.path.join(os.getcwd(), cpkt_path)
    model = Autoencoder.load_from_checkpoint(cpkt_path, 
                                            discriminator=Discriminator(), 
                                            generator=Generator(in_channels=2, out_channels=2), 
                                            visualize=False,
                                            alpha_penalty=10,
                                            alpha_fidelity=10,
                                            n_critic=10,
                                            d_learning_rate=1e-4,
                                            d_scheduler_step_size=1000,
                                            d_scheduler_gamma=1,
                                            g_learning_rate=1e-4,
                                            g_scheduler_step_size=1000,
                                            g_scheduler_gamma=1,
                                            weight_clip = False,
                                            weight_clip_value=0.5,
                                            logging_freq=5,
                                            batch_size=1)
    
    generator = model.generator
    discriminator = model.discriminator

    return generator, discriminator


def generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=True):
    all_rows = []

    with open(f'scores{model_path}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])
        vctk_clean_files = os.listdir(os.path.join(os.getcwd(), 'data/wav/test_clean_wav/'))

        for i in tqdm(range(len(clean_files))):
            clean_wav, _ = torchaudio.load(os.path.join(test_clean_path, clean_files[i]))
            noisy_wav, _ = torchaudio.load(os.path.join(test_noisy_path, noisy_files[i]))

            # Compute the STFT of the audio
            clean_stft = torch.stft(clean_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
            noisy_stft = torch.stft(noisy_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

            # Stack the real and imaginary parts of the STFT
            clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
            noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)

            # fake_clean_stft, mask = generator(noisy_stft)
            fake_clean_stft = noisy_stft

            real_clean_waveform = stft_to_waveform(clean_stft, device=torch.device('cpu')).detach()
            fake_clean_waveform = stft_to_waveform(fake_clean_stft, device=torch.device('cpu')).detach()

            if generator is not None:
                reference_index = random.choice([j for j in range(len(clean_files)) if j != i])
                non_matching_reference_waveform = torchaudio.load(os.path.join(test_clean_path, clean_files[reference_index]))[0]
            else:
                reference_index = random.choice(range(len(vctk_clean_files)))
                non_matching_reference_waveform = torchaudio.load(os.path.join(os.getcwd(), 'data/wav/test_clean_wav/', vctk_clean_files[reference_index]))[0]

            sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred = compute_scores(
                                                real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform, use_pesq=use_pesq)

            all_rows.append([sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred])
        
        ## Means
        writer.writerow(["Mean scores"])
        writer.writerow(np.mean(all_rows, axis=0))

        ## Standard errors of the means
        writer.writerow(["SE of the means" ])
        se_values = np.std(all_rows, axis=0) / np.sqrt(len(all_rows))
        writer.writerow(se_values)
        
        ## All scores
        writer.writerow(["All Scores"])
        for row in all_rows:
            writer.writerow(row)

def discriminator_scores(discriminator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path):
    all_rows = []
    with open(f'discriminator_scores{model_path}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Real", "Fake"])

        for i in tqdm(range(10)):
            clean_stft = torch.load(os.path.join(test_clean_path, clean_files[i])).requires_grad_(False)
            noisy_stft = torch.load(os.path.join(test_noisy_path, noisy_files[i])).requires_grad_(False)

            real_output = discriminator(clean_stft).mean()
            fake_output = discriminator(noisy_stft).mean()

            all_rows.append([real_output.item(), fake_output.item()])

        for row in all_rows:
            writer.writerow(row)

        
def main(model_path, get_generator_scores = True, get_discriminator_scores = False, limit_samples=False):
    val_dataset = AudioDataset(clean_path=test_clean_dir, noisy_path=test_noisy_dir, is_train=False, authentic=False, only_noisy=True)

    data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=20 if torch.cuda.is_available() else 1, 
                            persistent_workers=True, pin_memory=True, drop_last=True)
    model = Autoencoder.load_from_checkpoint(model_path)
    model.eval()

    trainer = Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        check_val_every_n_epoch=1,
        deterministic=True,
        )

    predictions = trainer.predict(model, data_loader)

    return predictions

    if model_path:
        generator, discriminator = load_model(model_path)
    test_clean_path = os.path.join(os.getcwd(), test_clean_dir)
    test_noisy_path = os.path.join(os.getcwd(), test_noisy_dir)
    if limit_samples:
        clean_files = os.listdir(test_clean_path)[:limit_samples]
        noisy_files = os.listdir(test_noisy_path)[:limit_samples]
    else:
        clean_files = os.listdir(test_clean_path)
        noisy_files = os.listdir(test_noisy_path)
    if get_generator_scores:
        if model_path:
            generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=use_pesq)
        else:
            generator_scores(None, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path='_noisy', use_pesq=use_pesq)
    if get_discriminator_scores:
        if model_path:
            discriminator_scores(discriminator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path)
        else:
            discriminator_scores(None, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path='_noisy')

if __name__ == '__main__':
    main(model_path = model_path, 
         get_generator_scores = get_generator_scores, 
         get_discriminator_scores = get_discriminator_scores, 
         limit_samples=limit_samples)
