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
use_pesq = False


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

        
def main(model_path):
    val_dataset = AudioDataset(clean_path=test_clean_dir, noisy_path=test_noisy_dir, is_train=False, authentic=False)
    data_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=20 if torch.cuda.is_available() else 1, 
                            persistent_workers=True, pin_memory=True, drop_last=True)
    model = Autoencoder.load_from_checkpoint(model_path)
    model.eval()
    trainer = Trainer(accelerator='gpu' if torch.cuda.is_available() else 'cpu',
                      check_val_every_n_epoch=1,
                      deterministic=True)
    predictions = trainer.predict(model, data_loader)

    # extract real_clean
    real_clean = [p[:][0][:][:][:][:] for p in predictions]
    # extract fake_clean and remove mask
    fake_clean = [p[:][1][0][:][:][:][:] for p in predictions] 

    clean_reference_filenames = os.listdir(os.path.join(os.getcwd(), 'data/wav/test_clean_wav/'))
    all_rows = []
    csv_name = model_path.split('/')[-1][:-5]
    with open(f'scores_{csv_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])
        for i in range(len(fake_clean)):

            reference_index = random.choice(range(len(clean_reference_filenames)))
            non_matching_reference_waveform = torchaudio.load(os.path.join(os.getcwd(), 'data/wav/test_clean_wav/', clean_reference_filenames[reference_index]))[0]

            sisnr_score = compute_scores(real_clean[i], fake_clean[i], non_matching_reference_waveform, use_pesq=use_pesq)
            # sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred = compute_scores(...)
            dnsmos_score = mos_squim_score = estoi_score = pesq_normal_score = pesq_torch_score = stoi_pred = pesq_pred = si_sdr_pred = 0
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


if __name__ == '__main__':
    main(model_path)
