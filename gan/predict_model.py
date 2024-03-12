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
from pesq import pesq
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from gan import stft_to_waveform
from speechmos import dnsmos
import os
import hydra
import numpy as np
from tqdm import tqdm
import csv
import random
import numpy as np
torch.set_grad_enabled(False)


@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):

    checkpoint_path = 'models/epoch=84.ckpt'
    test_clean_path = 'data/test_clean_stft/'
    test_noisy_path = 'data/test_noisy_stft/'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Autoencoder()
    checkpoint_path = os.path.join(hydra.utils.get_original_cwd(), checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))

    # # Load the trained model
    # model.load_state_dict(checkpoint['state_dict'])
    # # Get the generator
    # generator = model.generator
    
    generator = Generator()

    test_clean_path = os.path.join(hydra.utils.get_original_cwd(), test_clean_path)
    test_noisy_path = os.path.join(hydra.utils.get_original_cwd(), test_noisy_path)

    clean_files = [file for file in os.listdir(test_clean_path) if file.endswith('.pt')]
    noisy_files = [file for file in os.listdir(test_noisy_path) if file.endswith('.pt')]

    sisnr_scores = estoi_scores = mos_squim_scores = dnsmos_scores = pesq_normal_scores = pesq_torch_scores = np.array([])
    all_rows = []

    with open('scores.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch"])

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
            sisnr_scores = np.append(sisnr_scores, sisnr_score.item())

            ## Perceptual Evaluation of Speech Quality
            pesq_torch = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb').to(device)
            pesq_torch_score = pesq_torch(real_clean_waveform, fake_clean_waveform)
            pesq_torch_scores = np.append(pesq_torch_scores, pesq_torch_score.item())

            ## Perceptual Evaluation of Speech Quality
            pesq_normal_score = pesq(fs=16000, ref=real_clean_waveform.numpy(), deg=fake_clean_waveform.numpy(), mode='wb')
            pesq_normal_scores = np.append(pesq_normal_scores, pesq_normal_score)

            ## Deep Noise Suppression Mean Opinion Score (DNSMOS)
            dnsmos_score = dnsmos.run(fake_clean_waveform.numpy(), 16000)['ovrl_mos']
            dnsmos_scores = np.append(dnsmos_scores, dnsmos_score)

            ## MOS Squim
            reference_index = random.choice([j for j in range(len(clean_files)) if j != i])
            reference_waveform = torch.load(os.path.join(test_clean_path, clean_files[reference_index])).requires_grad_(False)
            reference_waveform = stft_to_waveform(reference_waveform.squeeze(0), device=torch.device('cpu')).detach()
            subjective_model = SQUIM_SUBJECTIVE.get_model()
            mos_squim_score = subjective_model(fake_clean_waveform.unsqueeze(0), reference_waveform)
            mos_squim_scores = np.append(mos_squim_scores, mos_squim_score.item())

            ## Extended Short Time Objective Intelligibility
            estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
            estoi_score = estoi(preds = fake_clean_waveform, target = real_clean_waveform)
            estoi_scores = np.append(estoi_scores, estoi_score.item())
 
            all_rows.append([sisnr_score.item(), dnsmos_score, mos_squim_score.item(), estoi_score.item(), pesq_normal_score, pesq_torch_score.item()])
        
        ## Mean scores
        writer.writerow(["Mean scores"])
        writer.writerow([np.mean(sisnr_scores), 
                         np.mean(dnsmos_scores), 
                         np.mean(mos_squim_scores), 
                         np.mean(estoi_scores), 
                         np.mean(pesq_normal_scores), 
                         np.mean(pesq_torch_scores)])


        ## Standard errors of the means
        writer.writerow(["SE of the means" ])
        writer.writerow([np.std(sisnr_scores)       /np.sqrt(len(sisnr_scores)), 
                         np.std(dnsmos_scores)      /np.sqrt(len(dnsmos_scores)), 
                         np.std(mos_squim_scores)   /np.sqrt(len(mos_squim_scores)), 
                         np.std(estoi_scores)       /np.sqrt(len(estoi_scores)), 
                         np.std(pesq_normal_scores) /np.sqrt(len(pesq_normal_scores)), 
                         np.std(pesq_torch_scores)  /np.sqrt(len(pesq_torch_scores))])
        
        writer.writerow(["All Scores"])


        for row in all_rows:
            writer.writerow(row)

        

            





if __name__ == '__main__':
    main()
