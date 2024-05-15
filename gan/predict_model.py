import torch
import torchaudio
from models.autoencoder import Autoencoder
from models.generator import Generator
from data.data_loader import AudioDataset, PreMadeDataset
from torch.utils.data import DataLoader
from utils.utils import stft_to_waveform, compute_scores
from pytorch_lightning import Trainer
import os
import numpy as np
import tqdm
import csv
import random
import numpy as np
import librosa.display
torch.set_grad_enabled(False)
from collections import OrderedDict
PYTORCH_ENABLE_MPS_FALLBACK=1
PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT=10

clean_path = 'data/test_clean_sampled_x'
noisy_path = 'data/test_noisy_sampled_x'

# fake_clean_path = 'data/AudioSet/fake_clean_triple_train'
# fake_clean_path = 'data/fake_clean_test_1000e_30_april_x' # if you want to use pre-generated samples or untouched noisy samples (no model)


# set model path to False if you don't want to generate new samples
model_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/models/swa_generator_epoch_129.ckpt'
fraction = 1.
csv_name = 'AudioSet_300e'
device = torch.device('mps')
authentic = True

### Metrics ###
use_sisnr=     True
use_dnsmos=    True
use_mos_squim= True
use_estoi=     True
use_pesq=      True
use_pred=      True
###############


def data_load():
    # dataset = PreMadeDataset(clean_path, noisy_path, fraction)
    dataset = AudioDataset(clean_path, noisy_path, is_train=False, fraction=fraction, authentic=authentic)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1 if torch.device == 'cpu' else 5, 
                            persistent_workers=True, pin_memory=True, drop_last=True)
    return data_loader

def model_load(model_path):
    try:
        autoencoder = Autoencoder.load_from_checkpoint(model_path, map_location=device)
        generator = autoencoder.generator
        discriminator = autoencoder.discriminator
        return autoencoder, generator, discriminator
    except:
        print("Could not load autoencoder, trying to load generator only")
        generator = Generator()
        checkpoint = torch.load(model_path, map_location=device)
        new_state_dict = OrderedDict()
        for k, v in checkpoint.items():
            name = k[7:] if k.startswith('module.') else k  # remove `module.` prefix if present
            if k != 'n_averaged':
                new_state_dict[name] = v
        generator.load_state_dict(new_state_dict)
        print("Generator loaded from checkpoint")
        return None, generator, None

def discriminator_scores(model_path, device='cuda'):
    data_loader = data_load()
    _, _, model = model_load(model_path)

    model.to(device)
    model.eval()

    with open('discriminator_scores.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["D_Real", "D_Fake"])
        for batch in tqdm.tqdm(data_loader):
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
        mos_reference_path = clean_path
        clean_reference_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), mos_reference_path)) if file.endswith('.wav')]
    all_rows = []
    with open(f'{csv_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])
        for i in tqdm.tqdm(range(len(fake_clean))):

            if use_mos_squim:
                reference_index = random.choice(range(len(clean_reference_filenames)))
                non_matching_reference_waveform = torchaudio.load(os.path.join(os.getcwd(), mos_reference_path, clean_reference_filenames[reference_index]))[0]
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
            
            all_rows.append([sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred])

        sisnr_scores, dnsmos_scores, mos_squim_scores, estoi_scores, pesq_normal_scores, pesq_torch_scores, stoi_preds, pesq_preds, si_sdr_preds = zip(*all_rows)
        pesq_normal_scores = [score for score in pesq_normal_scores if score != "Error"]
        stoi_preds = [score for score in stoi_preds if score > 0]
        pesq_preds = [score for score in pesq_preds if score > 0]
        si_sdr_preds = [score for score in si_sdr_preds if score > 0]
        
        ## Means
        writer.writerow(["Mean scores"])
        sisnr_mean = np.mean(sisnr_scores)
        dnsmos_mean = np.mean(dnsmos_scores)
        mos_squim_mean = np.mean(mos_squim_scores)
        estoi_mean = np.mean(estoi_scores)
        pesq_normal_mean = np.mean(pesq_normal_scores)
        pesq_torch_mean = np.mean(pesq_torch_scores)
        stoi_pred_mean = np.mean(stoi_preds)
        pesq_pred_mean = np.mean(pesq_preds)
        si_sdr_pred_mean = np.mean(si_sdr_preds)

        mean_scores = [sisnr_mean, dnsmos_mean, mos_squim_mean, estoi_mean, pesq_normal_mean, pesq_torch_mean, stoi_pred_mean, pesq_pred_mean, si_sdr_pred_mean]
        writer.writerow(mean_scores)

        ## Standard errors of the means
        writer.writerow(["Standard errors of the means"])
        sisnr_sem = np.std(sisnr_scores) / np.sqrt(len(sisnr_scores))
        dnsmos_sem = np.std(dnsmos_scores) / np.sqrt(len(dnsmos_scores))
        mos_squim_sem = np.std(mos_squim_scores) / np.sqrt(len(mos_squim_scores))
        estoi_sem = np.std(estoi_scores) / np.sqrt(len(estoi_scores))
        pesq_normal_sem = np.std(pesq_normal_scores) / np.sqrt(len(pesq_normal_scores))
        pesq_torch_sem = np.std(pesq_torch_scores) / np.sqrt(len(pesq_torch_scores))
        stoi_pred_sem = np.std(stoi_preds) / np.sqrt(len(stoi_preds))
        pesq_pred_sem = np.std(pesq_preds) / np.sqrt(len(pesq_preds))
        si_sdr_pred_sem = np.std(si_sdr_preds) / np.sqrt(len(si_sdr_preds))

        sem_scores = [sisnr_sem, dnsmos_sem, mos_squim_sem, estoi_sem, pesq_normal_sem, pesq_torch_sem, stoi_pred_sem, pesq_pred_sem, si_sdr_pred_sem]
        writer.writerow(sem_scores)


        
        ## All scores
        writer.writerow(["All Scores"])
        for row in all_rows:
            writer.writerow(row)


def generator_scores_model_sampled_clean_noisy(model_path):
    autoencoder, generator, discriminator = model_load(model_path)
    generator
    generator.eval()

    noisy_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), noisy_path)) if file.endswith('.wav')]
    noisy_files = [torchaudio.load(os.path.join(os.getcwd(), noisy_path, file))[0] for file in noisy_filenames]
    real_clean_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), clean_path)) if file.endswith('.wav')]
    clean_files = [torchaudio.load(os.path.join(os.getcwd(), clean_path, file))[0] for file in real_clean_filenames]

    if use_mos_squim:
        mos_reference_path = clean_path
        clean_reference_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), mos_reference_path)) if file.endswith('.wav')]
    all_rows = []
    with open(f'{csv_name}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])

        for i in tqdm.tqdm(range(len(noisy_files))):
            noisy_waveform = noisy_files[i]
            # Compute the STFT of the audio
            noisy_file = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
            # Stack the real and imaginary parts of the STFT
            noisy_file = torch.stack((noisy_file.real, noisy_file.imag), dim=1).to(device)
            fake_clean = generator(noisy_file)
            fake_clean = stft_to_waveform(fake_clean[0].to(torch.device('cpu')), device = 'cpu')


            if use_mos_squim:
                reference_index = random.choice(range(len(clean_reference_filenames)))
                non_matching_reference_waveform = torchaudio.load(os.path.join(os.getcwd(), mos_reference_path, clean_reference_filenames[reference_index]))[0]
            else: 
                non_matching_reference_waveform = None

            sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred = compute_scores(
                                                                                                clean_files[i], fake_clean, non_matching_reference_waveform,
                                            use_sisnr=     use_sisnr, 
                                            use_dnsmos=    use_dnsmos, 
                                            use_mos_squim= use_mos_squim, 
                                            use_estoi=     use_estoi,
                                            use_pesq=      use_pesq, 
                                            use_pred=      use_pred)
            
            all_rows.append([sisnr_score, dnsmos_score, mos_squim_score, estoi_score, pesq_normal_score, pesq_torch_score, stoi_pred, pesq_pred, si_sdr_pred])

        sisnr_scores, dnsmos_scores, mos_squim_scores, estoi_scores, pesq_normal_scores, pesq_torch_scores, stoi_preds, pesq_preds, si_sdr_preds = zip(*all_rows)
        pesq_normal_scores = [score for score in pesq_normal_scores if score != "Error"]
        stoi_preds = [score for score in stoi_preds if score > 0]
        pesq_preds = [score for score in pesq_preds if score > 0]
        si_sdr_preds = [score for score in si_sdr_preds if score > 0]
        
        ## Means
        writer.writerow(["Mean scores"])
        sisnr_mean = np.mean(sisnr_scores)
        dnsmos_mean = np.mean(dnsmos_scores)
        mos_squim_mean = np.mean(mos_squim_scores)
        estoi_mean = np.mean(estoi_scores)
        pesq_normal_mean = np.mean(pesq_normal_scores)
        pesq_torch_mean = np.mean(pesq_torch_scores)
        stoi_pred_mean = np.mean(stoi_preds)
        pesq_pred_mean = np.mean(pesq_preds)
        si_sdr_pred_mean = np.mean(si_sdr_preds)

        mean_scores = [sisnr_mean, dnsmos_mean, mos_squim_mean, estoi_mean, pesq_normal_mean, pesq_torch_mean, stoi_pred_mean, pesq_pred_mean, si_sdr_pred_mean]
        writer.writerow(mean_scores)

        ## Standard errors of the means
        writer.writerow(["Standard errors of the means"])
        sisnr_sem = np.std(sisnr_scores) / np.sqrt(len(sisnr_scores))
        dnsmos_sem = np.std(dnsmos_scores) / np.sqrt(len(dnsmos_scores))
        mos_squim_sem = np.std(mos_squim_scores) / np.sqrt(len(mos_squim_scores))
        estoi_sem = np.std(estoi_scores) / np.sqrt(len(estoi_scores))
        pesq_normal_sem = np.std(pesq_normal_scores) / np.sqrt(len(pesq_normal_scores))
        pesq_torch_sem = np.std(pesq_torch_scores) / np.sqrt(len(pesq_torch_scores))
        stoi_pred_sem = np.std(stoi_preds) / np.sqrt(len(stoi_preds))
        pesq_pred_sem = np.std(pesq_preds) / np.sqrt(len(pesq_preds))
        si_sdr_pred_sem = np.std(si_sdr_preds) / np.sqrt(len(si_sdr_preds))

        sem_scores = [sisnr_sem, dnsmos_sem, mos_squim_sem, estoi_sem, pesq_normal_sem, pesq_torch_sem, stoi_pred_sem, pesq_pred_sem, si_sdr_pred_sem]
        writer.writerow(sem_scores)
        
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

def generate_fake_clean(model_path):
    #### from Waveform files ####
    autoencoder, generator, discriminator = model_load(model_path)
    noisy_filenames = sorted([file for file in os.listdir(os.path.join(os.getcwd(), noisy_path)) if file.endswith('.wav')])
    noisy_files = [torchaudio.load(os.path.join(os.getcwd(), noisy_path, file))[0] for file in noisy_filenames]

    clean_filenames = sorted([file for file in os.listdir(os.path.join(os.getcwd(), clean_path)) if file.endswith('.wav')])
    clean_files = [torchaudio.load(os.path.join(os.getcwd(), clean_path, file))[0] for file in clean_filenames]

    for i, noisy_file in tqdm.tqdm(enumerate(noisy_files)):
        noisy_waveform = noisy_file
        clean_waveform = clean_files[i]
        # Compute the STFT of the audio
        noisy_file = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
        clean_file = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

        # Stack the real and imaginary parts of the STFT
        noisy_file = torch.stack((noisy_file.real, noisy_file.imag), dim=1)
        clean_file = torch.stack((clean_file.real, clean_file.imag), dim=1)

        input_file = noisy_file - clean_file

        fake_clean = generator(input_file)
        fake_clean = stft_to_waveform(fake_clean[0], device = device)
        torchaudio.save(f'/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/fake_clean_triple_train/{noisy_filenames[i]}', fake_clean, 16000)

    # #### from STFT files ####
    # autoencoder, generator, discriminator = model_load(model_path)
    # noisy_filenames = [file for file in os.listdir(os.path.join(os.getcwd(), noisy_path)) if file.endswith('.pt')]
    # noisy_files = [torch.load(os.path.join(os.getcwd(), noisy_path, file)) for file in noisy_filenames]
    # for i, noisy_file in tqdm.tqdm(enumerate(noisy_files)):
    #     fake_clean = generator(noisy_file)
    #     fake_clean = stft_to_waveform(fake_clean[0], device = device)
    #     torchaudio.save(f'/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/fake_clean_test_800e_y/{noisy_filenames[i][:-7]}.wav', fake_clean, 16000)


if __name__ == '__main__':
    # generator_scores(model_path)
    # generate_fake_clean(model_path)
    generator_scores_model_sampled_clean_noisy(model_path)




    # _, generator, _ = model_load(model_paths)
    # input_waveform, sr = torchaudio.load('data/test_noisy_sampled/p232_001.wav')
    # # Resample to 16kHz
    # input = torchaudio.transforms.Resample(sr, 16000)(input_waveform)

    # # Transform to STFT
    # input = torch.stft(input, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)

    # input = torch.stack([input.real, input.imag], dim=1)


    # feature_maps = visualize_feature_maps(generator, input)
    # print("Feature map shapes:")
    # for i, feature_map in enumerate(feature_maps):
    #     print(f"Layer {i}: {feature_map.shape}")

    # # Visualize the feature maps
    # import matplotlib.pyplot as plt
    # fig, axs = plt.subplots(1, len(feature_maps) + 2, figsize=(20, 5))
    # # Visualize the input spectrogram with librosa
    # input_waveform = input_waveform.numpy()
    # input_waveform = librosa.resample(input_waveform, sr, 16000)
    # input_spectrogram = librosa.feature.melspectrogram(input_waveform, sr=16000, n_fft=512, hop_length=100, n_mels=80)
    # input_spectrogram = librosa.power_to_db(input_spectrogram, ref=np.max)
    # librosa.display.specshow(input_spectrogram.squeeze(0), ax=axs[0], y_axis='mel', x_axis='time')
    # axs[0].set_title("Input")

    # for i, feature_map in enumerate(feature_maps):
    #     ax = axs[i + 1]
    #     ax.imshow(feature_map.cpu().numpy())
    #     ax.set_title(f"Layer {i}")

    # # Visualize the output waveform
    # output_waveform = stft_to_waveform(generator(input)[0], device = 'cpu')
    # output_spectrogram = librosa.feature.melspectrogram(output_waveform.squeeze().numpy(), sr=16000, n_fft=512, hop_length=100, n_mels=80)
    # librosa.display.specshow(librosa.power_to_db(output_spectrogram, ref=np.max), ax=axs[-1], y_axis='mel', x_axis='time')
    # axs[-1].set_title("Output")

        
    # plt.show()
