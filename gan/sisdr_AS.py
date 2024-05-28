import torch
import torchaudio
from models.autoencoder import Autoencoder
from models.generator import Generator
from utils.utils import stft_to_waveform, compute_scores
from pytorch_lightning import Trainer
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import numpy as np
import tqdm
import csv
import random
import numpy as np
torch.set_grad_enabled(False)
from collections import OrderedDict
import librosa
import soundfile as sf
from scipy.io.wavfile import write
from torchaudio.pipelines import SQUIM_OBJECTIVE

clean_path = 'data/test_clean_sampled_x'
noisy_path = '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/test_sampled'

# fake_clean_path = 'data/AudioSet/fake_clean_triple_train'
# fake_clean_path = 'data/fake_clean_test_1000e_30_april_x' # if you want to use pre-generated samples or untouched noisy samples (no model)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# set model path to False if you don't want to generate new samples
model_path = '/Users/fredmac/Downloads/bachelor_project/models/Audioset/used_for_results/AudioSet_epoch=999.ckpt'
csv_name = 'dnsmos_noisy_AS'

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

def test_scores():
    autoencoder, generator, discriminator = model_load(model_path)
    generator.eval()

    noisy_filenames = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])
    with open(f'{csv_name}_{model_path.split("/")[-1]}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["dnsmos"])
        all_rows = []
        for noisy_file in tqdm.tqdm(noisy_filenames, desc='Generating and saving audio files'):
            # Load noisy waveform
            noisy_waveform, sr = torchaudio.load(os.path.join(noisy_path, noisy_file))
            
            # Compute the STFT of the noisy audio
            noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400), return_complex=True)
            
            # Stack the real and imaginary parts of the STFT
            noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1).to(device)
            
            # Generate clean waveform using the generator
            fake_clean_stft = generator(noisy_stft)
            fake_clean_waveform = stft_to_waveform(fake_clean_stft[0].to('cpu'), device='cpu')
            

            # # Save the generated clean waveform
            # save_path = '/Users/fredmac/Downloads/bachelor_project/data/AudioSet/fake_clean_AudioSet_model/'

            # torchaudio.save(save_path + noisy_file, fake_clean_waveform, sr)

            # objective_model = SQUIM_OBJECTIVE.get_model()
            # stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveform)
            # si_sdr_pred = si_sdr_pred.item()
            # all_rows.append([si_sdr_pred])
            
            dnsmos_input = noisy_waveform

            from speechmos import dnsmos
            dnsmos_input = dnsmos_input.numpy()
            if np.max(abs(dnsmos_input)) > 1:
                dnsmos_input = dnsmos_input / np.max(abs(dnsmos_input))
            dnsmos_score = dnsmos.run(dnsmos_input.squeeze(), 16000)['p808_mos']
            all_rows.append([dnsmos_score])
        
        mean = np.mean([row[0] for row in all_rows])
        sem = np.std([row[0] for row in all_rows]) / np.sqrt(len(all_rows))
        writer.writerow(["Mean scores"])
        writer.writerow([mean])
        writer.writerow(["Standard errors of the means"])
        writer.writerow([sem])
        for row in all_rows:
            writer.writerow(row)

if __name__ == "__main__":
    test_scores()
