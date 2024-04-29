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
import librosa.display
torch.set_grad_enabled(False)
PYTORCH_ENABLE_MPS_FALLBACK=1

clean_path = 'data/test_clean_sampled'
noisy_path = 'data/test_noisy_sampled'
# use fake clean path if you want to use pre-generated samples or untouched noisy samples (no model)
fake_clean_path = 'data/test_noisy_sampled'
model_paths = 'models/standardmodel1000.ckpt'
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

def visualize_feature_maps(model, input):
    # Visualize the feature maps of the model
    # Put input through the model and save every layer output
    feature_maps = []
    with torch.no_grad():
        _, _, maps = model(input)
        
        # Take the mean of the channel dimension
        for layer in maps:
            layer = layer.squeeze(0)
            feature_maps.append(layer.mean(dim=0))

    return feature_maps

if __name__ == '__main__':
    _, generator, _ = model_load(model_paths)
    input_waveform, sr = torchaudio.load('data/test_noisy_sampled/p232_001.wav')
    # Resample to 16kHz
    input = torchaudio.transforms.Resample(sr, 16000)(input_waveform)

    # Transform to STFT
    input = torch.stft(input, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

    input = torch.stack([input.real, input.imag], dim=1)


    feature_maps = visualize_feature_maps(generator, input)
    print("Feature map shapes:")
    for i, feature_map in enumerate(feature_maps):
        print(f"Layer {i}: {feature_map.shape}")

    # Visualize the feature maps
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, len(feature_maps) + 2, figsize=(20, 5))
    # Visualize the input spectrogram with librosa
    input_waveform = input_waveform.numpy()
    input_waveform = librosa.resample(input_waveform, sr, 16000)
    input_spectrogram = librosa.feature.melspectrogram(input_waveform, sr=16000, n_fft=512, hop_length=100, n_mels=80)
    input_spectrogram = librosa.power_to_db(input_spectrogram, ref=np.max)
    librosa.display.specshow(input_spectrogram.squeeze(0), ax=axs[0], y_axis='mel', x_axis='time')
    axs[0].set_title("Input")

    for i, feature_map in enumerate(feature_maps):
        ax = axs[i + 1]
        ax.imshow(feature_map.cpu().numpy())
        ax.set_title(f"Layer {i}")

    # Visualize the output waveform
    output_waveform = stft_to_waveform(generator(input)[0], device = 'cpu')
    output_spectrogram = librosa.feature.melspectrogram(output_waveform.squeeze().numpy(), sr=16000, n_fft=512, hop_length=100, n_mels=80)
    librosa.display.specshow(librosa.power_to_db(output_spectrogram, ref=np.max), ax=axs[-1], y_axis='mel', x_axis='time')
    axs[-1].set_title("Output")

        
    plt.show()
