import torch
from models.generator import Generator
from models.discriminator import Discriminator
from models.autoencoder import Autoencoder
from utils.utils import stft_to_waveform, compute_scores
import os
import hydra
import numpy as np
from tqdm import tqdm
import csv
import random
import numpy as np
torch.set_grad_enabled(False)

model_path = False#"models/10pct_889.ckpt"
get_generator_scores = True
get_discriminator_scores = False
limit_samples = 100     # False: use all samples, Integer: use only the first n samples
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
                                            batch_size=4)
    
    generator = model.generator
    discriminator = model.discriminator

    return generator, discriminator


def generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=True):
    all_rows = []

    with open(f'scores{model_path}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])

        for i in tqdm(range(len(clean_files))):
            clean_stft = torch.load(os.path.join(test_clean_path, clean_files[i])).requires_grad_(False)
            noisy_stft = torch.load(os.path.join(test_noisy_path, noisy_files[i])).requires_grad_(False)

            # fake_clean_stft, mask = generator(noisy_stft)
            fake_clean_stft = noisy_stft

            real_clean_waveform = stft_to_waveform(clean_stft, device=torch.device('cpu')).detach()
            fake_clean_waveform = stft_to_waveform(fake_clean_stft, device=torch.device('cpu')).detach()

            reference_index = random.choice([j for j in range(len(clean_files)) if j != i])
            non_matching_reference_stft = torch.load(os.path.join(test_clean_path, clean_files[reference_index])).requires_grad_(False)
            non_matching_reference_waveform = stft_to_waveform(non_matching_reference_stft.squeeze(0), device=torch.device('cpu')).detach()
            
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
    
    generator, discriminator = load_model(model_path)
    test_clean_path = os.path.join(os.getcwd(), 'data/test_clean_stft/')
    test_noisy_path = os.path.join(os.getcwd(), 'data/test_noisy_stft/')
    if limit_samples:
        clean_files = os.listdir(test_clean_path)[:limit_samples]
        noisy_files = os.listdir(test_noisy_path)[:limit_samples]
    else:
        clean_files = os.listdir(test_clean_path)
        noisy_files = os.listdir(test_noisy_path)
    if get_generator_scores:
        generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=use_pesq)
    if get_discriminator_scores:
        discriminator_scores(discriminator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path)

if __name__ == '__main__':
    main(model_path = model_path, 
         get_generator_scores = get_generator_scores, 
         get_discriminator_scores = get_discriminator_scores, 
         limit_samples=limit_samples)
