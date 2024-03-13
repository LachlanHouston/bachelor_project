import os
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from torchaudio.pipelines import SQUIM_SUBJECTIVE
from pesq import pesq
from speechmos import dnsmos
from gan.utils.utils import stft_to_waveform
import random
import tqdm
import numpy as np



def baseline_model():
    # Load test data
    test_clean_path = 'data/wav/test_clean_wav/'
    test_noisy_path = 'data/wav/test_noisy_wav/'
    test_clean_filenames = [file for file in os.listdir(test_clean_path) if file.endswith('.wav')]
    test_noisy_filenames = [file for file in os.listdir(test_noisy_path) if file.endswith('.wav')]

    test_clean_waveforms = []
    test_noisy_waveforms = []

    print(len(test_noisy_filenames))
    print(len(test_clean_filenames))


    for i in tqdm.tqdm(range(len(test_clean_filenames)), "load test"):
        test_clean_waveforms.append(torchaudio.load(test_clean_path + test_clean_filenames[i])[0].squeeze(0))
        test_noisy_waveforms.append(torchaudio.load(test_noisy_path + test_noisy_filenames[i])[0].squeeze(0))

    # Load training data
    train_clean_path = 'data/wav/clean_wav/'
    train_noisy_path = 'data/wav/noisy_wav/'
    train_clean_filenames = [file for file in os.listdir(train_clean_path) if file.endswith('.wav')]
    train_noisy_filenames = [file for file in os.listdir(train_noisy_path) if file.endswith('.wav')]

    train_clean_waveforms = []
    train_noisy_waveforms = []

    for i in tqdm.tqdm(range(len(train_clean_filenames)), "load train"):
        train_clean_waveforms.append(torchaudio.load(train_clean_path + train_clean_filenames[i])[0].squeeze(0))
        train_noisy_waveforms.append(torchaudio.load(train_noisy_path + train_noisy_filenames[i])[0].squeeze(0))


    # Create the mean mask based on training data
    mean_mask = torch.zeros(train_clean_waveforms[0].size())
    for i in tqdm.tqdm(range(len(train_clean_waveforms)), "compute mask"):
        mean_mask += train_clean_waveforms[i] - train_noisy_waveforms[i]
    mean_mask /= len(train_clean_waveforms)

    # Apply the mask to the test noisy data
    for i in range(len(test_noisy_waveforms)):
        test_noisy_waveforms[i] += mean_mask

    fake_clean_test_waveforms = test_noisy_waveforms

    # Save the waveforms
    # for i in range(len(fake_clean_test_waveforms)):
    #     torchaudio.save('data/test_baseline/' + test_noisy_filenames[i], fake_clean_test_waveforms[i], 16000)

    # Turn into torch tensors
    test_clean_waveforms = torch.stack(test_clean_waveforms)
    fake_clean_test_waveforms = torch.stack(fake_clean_test_waveforms)

#%%

    # Compute SNR
    snr = ScaleInvariantSignalNoiseRatio()
    snr_val = snr(preds = fake_clean_test_waveforms, target = test_clean_waveforms, )
    print('SNR:', snr_val)

    # Compute eSTOI 
    estoi = ShortTimeObjectiveIntelligibility(16000, extended=True)
    estoi_val = estoi(preds = fake_clean_test_waveforms, target = test_clean_waveforms)
    print('eSTOI:', estoi_val)

    ## MOS Squim
    mos_squim_scores = np.array([])
    for fake_clean_waveform in tqdm.tqdm(fake_clean_test_waveforms, "MOS Squim"):
        reference_index = random.choice([j for j in range(len(test_clean_waveforms)) if j != i])
        reference_waveform = torchaudio.load(os.path.join(test_clean_path, test_clean_filenames[reference_index]))[0].squeeze(0).requires_grad_(False)
        subjective_model = SQUIM_SUBJECTIVE.get_model()
        mos_squim_score = subjective_model(fake_clean_waveform.unsqueeze(0), reference_waveform.unsqueeze(0))
        mos_squim_scores = np.append(mos_squim_scores, mos_squim_score.item())
    print('MOS Squim:', mos_squim_scores.mean())

    # Compute DNSMOS
    dnsmos_val = []
    for i in tqdm.tqdm(range(len(test_clean_waveforms)), "DNSMOS"):
        dnsmos_val.append(dnsmos.run(fake_clean_test_waveforms[i].numpy(), 16000)['ovrl_mos'])
    print('DNSMOS:', sum(dnsmos_val) / len(dnsmos_val))

    # Compute PESQ
    pesq_model = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
    pesq_torch = pesq_model(test_clean_waveforms, fake_clean_test_waveforms)
    print('PESQ Torch:', pesq_torch)

    pesq_normal_scores = [pesq(fs=16000, ref=test_clean_waveforms[i].numpy(), deg=fake_clean_test_waveforms[i].numpy(), mode='wb') for i in range(len(test_clean_waveforms))]
    pesq_normal_score = sum(pesq_normal_scores) / len(pesq_normal_scores)
    print('PESQ Normal:', pesq_normal_score)
    
if __name__ == '__main__':
    baseline_model()

    # SNR: 8.3789
    # eSTOI: 0.6115
    # SegSNR: 4.7741
    # DNSMOS: 2.3418
