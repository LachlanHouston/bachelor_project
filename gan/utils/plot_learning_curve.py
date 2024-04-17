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
from gan.utils.utils import stft_to_waveform
from tqdm import tqdm

PYTORCH_CUDA_ALLOC_CONF=True

def SNR_scores(train=True, model=None):
    clean_train_path = os.path.join(os.getcwd(), 'data/clean_raw/')
    noisy_train_path = os.path.join(os.getcwd(), 'data/noisy_raw/')
    clean_test_path = os.path.join(os.getcwd(), 'data/test_clean_raw/')
    noisy_test_path = os.path.join(os.getcwd(), 'data/test_noisy_raw/')

    # clean_train_files = sorted(os.listdir(clean_train_path))
    # noisy_train_files = sorted(os.listdir(noisy_train_path))
    clean_test_files = sorted(os.listdir(clean_test_path))
    noisy_test_files = sorted(os.listdir(noisy_test_path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    if model is None:
        model = Autoencoder(alpha_penalty =         10,
                            alpha_fidelity =        10,

                            n_critic =              10,
                            use_bias =              True,
                            
                            d_learning_rate =       1e-4,
                            g_learning_rate =       1e-4,

                            weight_clip =           False,
                            weight_clip_value =     1,

                            visualize =             True,
                            logging_freq =          5,
                            log_all_scores =        False,
                            batch_size =            4,
                            L2_reg =                False,
                            sisnr_loss =            False,
                            val_fraction =          1.,
                            dataset =               "VCTK"
                        ).generator

    model.to(device)

    snr_scores = []

    print("Test data")
    for i in range(len(clean_test_files)):
        print(f"File {i}/{len(clean_test_files)}")
        print(f"File: {clean_test_files[i]}")

        # Load data
        clean_waveform, sr = torchaudio.load(os.path.join(clean_test_path, clean_test_files[i]))
        noisy_waveform, _ = torchaudio.load(os.path.join(noisy_test_path, noisy_test_files[i]))

        # Resample to 16kHz
        # clean_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(clean_waveform).to(device)
        # noisy_waveform = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(noisy_waveform).to(device)

        # Transform waveform to STFT
        clean_waveform = clean_waveform.to(device)
        noisy_waveform = noisy_waveform.to(device)

        clean_stft = torch.stft(clean_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)
        noisy_stft = torch.stft(noisy_waveform, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)

        clean_stft = torch.stack([clean_stft.real, clean_stft.imag], dim=1)
        noisy_stft = torch.stack([noisy_stft.real, noisy_stft.imag], dim=1)
        
        output_real, _ = model(clean_stft)
        output_fake, _ = model(noisy_stft)

        # Transform the output to waveform
        real_clean_waveform = stft_to_waveform(output_real, device=device)
        fake_clean_waveform = stft_to_waveform(output_fake, device=device)

        # Compute SI-SNR
        sisnr = ScaleInvariantSignalNoiseRatio().to(device)
        snr = sisnr(preds=fake_clean_waveform, target=real_clean_waveform)
        print(f"SNR: {snr}")

        snr_scores.append(snr)

        # Clean up
        del clean_waveform, noisy_waveform, clean_stft, noisy_stft, output_real, output_fake, real_clean_waveform, fake_clean_waveform

        # Clear cache
        torch.cuda.empty_cache()

    return snr_scores

def compute_scores(real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform, use_pesq=True):

    if real_clean_waveform.size() == (1, 32000):
        real_clean_waveform = real_clean_waveform.squeeze(0)
    if fake_clean_waveform.size() == (1, 32000):
        fake_clean_waveform = fake_clean_waveform.squeeze(0)

    ## SI-SNR
    sisnr = ScaleInvariantSignalNoiseRatio().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    sisnr_score = sisnr(preds=fake_clean_waveform, target=real_clean_waveform)

    # if use_pesq:
    #     from pesq import pesq
    #     ## PESQ Normal
    #     pesq_normal_score = pesq(fs=16000, ref=real_clean_waveform.numpy(), deg=fake_clean_waveform.numpy(), mode='wb')

    #     ## PESQ Torch
    #     pesq_torch = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
    #     pesq_torch_score = pesq_torch(real_clean_waveform, fake_clean_waveform)

    return sisnr_score.item()

def generator_scores(generator, test_clean_path, test_noisy_path, clean_files, noisy_files, model_path, use_pesq=True):
    all_rows = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    generator = generator.to(device)

    with open(f'scores{model_path}.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["SI-SNR", "DNSMOS", "MOS Squim", "eSTOI", "PESQ", "PESQ Torch", "STOI pred", "PESQ pred", "SI-SDR pred"])

        for i in tqdm(range(len(clean_files))):
            clean_wav, sr = torchaudio.load(os.path.join(test_clean_path, clean_files[i]))
            noisy_wav, _ = torchaudio.load(os.path.join(test_noisy_path, noisy_files[i]))

            # Resample to 16kHz
            clean_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(clean_wav).to(device)
            noisy_wav = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)(noisy_wav).to(device)

            # Compute the STFT of the audio
            clean_stft = torch.stft(clean_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)
            noisy_stft = torch.stft(noisy_wav, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device), return_complex=True)

            # Stack the real and imaginary parts of the STFT
            clean_stft = torch.stack((clean_stft.real, clean_stft.imag), dim=1)
            noisy_stft = torch.stack((noisy_stft.real, noisy_stft.imag), dim=1)

            fake_clean_stft, _ = generator(noisy_stft)

            real_clean_waveform = stft_to_waveform(clean_stft, device=device).detach()
            fake_clean_waveform = stft_to_waveform(fake_clean_stft, device=device).detach()

            sisnr_score = compute_scores(
                                                real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform = 1, use_pesq=use_pesq)

            all_rows.append([sisnr_score])
        
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
    model_names = ['10p.ckpt']#, '20_cpkt', '30_cpkt', '40_cpkt', '50_cpkt', '60_cpkt', '70_cpkt', '80_cpkt', '90_cpkt', '100_cpkt']

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
        
        SNR_train = generator_scores(model, os.path.join(os.getcwd(), 'data/clean_raw/'), os.path.join(os.getcwd(), 'data/noisy_raw/'), clean_files, noisy_files, model_name + "_train", use_pesq=True)
        SNR_test = generator_scores(model, os.path.join(os.getcwd(), 'data/test_clean_raw/'), os.path.join(os.getcwd(), 'data/test_noisy_raw/'), test_clean_files, test_noisy_files, model_name + "_test", use_pesq=True)





