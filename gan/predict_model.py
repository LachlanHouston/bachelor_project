import torch
import torchaudio
from models.autoencoder import Autoencoder
from data.data_loader import AudioDataset, PreMadeDataset
from torch.utils.data import DataLoader
from utils.utils import stft_to_waveform, compute_scores
from pytorch_lightning import Trainer
import os
import numpy as np
from tqdm import tqdm
import csv
import random
import numpy as np
torch.set_grad_enabled(False)
PYTORCH_ENABLE_MPS_FALLBACK=1

clean_path = 'data/test_clean_sampled'
noisy_path = 'data/test_noisy_sampled'
# use fake clean path if you want to use pre-generated samples or untouched noisy samples (no model)
fake_clean_path = 'data/test_noisy_sampled'
model_paths = [False] #[f"models/learning_curve_500epochs/{pct}p.ckpt" for pct in [90,100]]
fraction = 1.
device = torch.device('mps')

### Metrics ###
use_sisnr=     True
use_dnsmos=    False
use_mos_squim= False
use_estoi=     False
use_pesq=      False
use_pred=      False
###############


def data_load():
    # dataset = PreMadeDataset(clean_path, noisy_path, fraction)
    dataset = AudioDataset(clean_path, noisy_path, is_train=False, fraction=fraction, authentic=True)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1 if torch.device == 'cpu' else 5, 
                            persistent_workers=True, pin_memory=True, drop_last=True)
    return data_loader

def model_load(model_path):
    autoencoder = Autoencoder.load_from_checkpoint(model_path)
    generator = autoencoder.generator
    discriminator = autoencoder.discriminator
    return autoencoder, generator, discriminator

def discriminator_scores(model_path, device='cuda'):
    data_loader = data_load()
    _, _, model = model_load(model_path)

    model.to(device)
    model.eval()

    with open('discriminator_scores.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["D_Real", "D_Fake"])
        for batch in tqdm(data_loader):
            real_clean = batch[0].squeeze(1).to(device)
            real_noisy = batch[1].squeeze(1).to(device)
            D_clean = model(real_clean)
            D_noisy = model(real_noisy)
            writer.writerow([D_clean.item(), D_noisy.item()])

def generator_scores(model_path):
    if model_path:
        model, _, _ = model_load(model_path)
        data_loader = data_load()

        model.to(device)
        model.eval()

        trainer = Trainer(accelerator='gpu' if device != torch.device('cpu') else 'cpu',
                        check_val_every_n_epoch=1,
                        deterministic=True)
        predictions = trainer.predict(model, data_loader)
        # extract real_clean
        real_clean = [p[:][0][:][:][:][:] for p in predictions]
        # extract fake_clean and remove mask
        fake_clean = [p[:][1][0][:][:][:][:] for p in predictions]
    
        real_clean = [stft_to_waveform(stft, device = device) for stft in real_clean]
        fake_clean = [stft_to_waveform(stft, device = device) for stft in fake_clean]

    else:
        fake_clean_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), fake_clean_path)) if file.endswith('.wav')]
        fake_clean = [torchaudio.load(os.path.join(os.getcwd(), fake_clean_path, file))[0] for file in fake_clean_filenames]
        real_clean_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), clean_path)) if file.endswith('.wav')]
        real_clean = [torchaudio.load(os.path.join(os.getcwd(), clean_path, file))[0] for file in real_clean_filenames]

    if use_mos_squim:
        clean_reference_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), 'data/test_clean_sampled/')) if file.endswith('.wav')]
    all_rows = []
    csv_name = model_path.split('/')[-1][:-5] if model_path else 'no_model'
    with open(f'sisnr_test_{csv_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])
        for i in tqdm(range(len(fake_clean))):

            if use_mos_squim:
                reference_index = random.choice(range(len(clean_reference_filenames)))
                non_matching_reference_waveform = torchaudio.load(os.path.join(os.getcwd(), 'data/test_clean_sampled/', clean_reference_filenames[reference_index]))[0]
            else: 
                non_matching_reference_waveform = None
            sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred = compute_scores(
                                                                                                real_clean[i], fake_clean[i], non_matching_reference_waveform, 
                                         use_sisnr=     use_sisnr, 
                                         use_dnsmos=    use_dnsmos, 
                                         use_mos_squim= use_mos_squim, 
                                         use_estoi=     use_estoi,
                                         use_pesq=      use_pesq, 
                                         use_pred=      use_pred)
            
            # if not stoi_pred > 0:
            #     print("pred is NaN. Skipping...")
            #     print('{stoi_pred}')
            #     continue
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
    for model_path in model_paths:
        print(f"Starting with {model_path}")
        generator_scores(model_path)# if torch.cuda.is_available() else 'cpu')
        print(f"Done with {model_path}")
    # discriminator_scores(model_path, device='cuda' if torch.cuda.is_available() else 'cpu')
    print("Done")
