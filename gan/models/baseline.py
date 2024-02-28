import os
import torch
import torchaudio
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchmetrics.audio import PerceptualEvaluationSpeechQuality
from speechmos import dnsmos
from gan.utils.utils import SegSNR


def baseline_model():
    # Load all test data
    test_clean_path = 'data/test_clean_wav/'
    test_noisy_path = 'data/test_noisy_wav/'
    clean_filenames = os.listdir(test_clean_path)
    noisy_filenames = os.listdir(test_noisy_path)

    clean_waveforms = []
    noisy_waveforms = []

    for i in range(len(clean_filenames)):
        if clean_filenames[i].endswith('.wav'):
            clean_waveforms.append(torchaudio.load(test_clean_path + clean_filenames[i])[0].squeeze(0))
        if noisy_filenames[i].endswith('.wav'):
            noisy_waveforms.append(torchaudio.load(test_noisy_path + noisy_filenames[i])[0].squeeze(0))
    
    # # Find largest waveform
    # max_length = 0
    # for waveform in clean_waveforms:
    #     if waveform.size(1) > max_length:
    #         max_length = waveform.size(1)

    # # Pad all waveforms to the same length
    # for i in range(len(clean_waveforms)):
    #     clean_waveforms[i] = torch.nn.functional.pad(clean_waveforms[i], (0, max_length - clean_waveforms[i].size(1)))
    #     noisy_waveforms[i] = torch.nn.functional.pad(noisy_waveforms[i], (0, max_length - noisy_waveforms[i].size(1)))

    # Create the mean mask
    mean_mask = torch.zeros(clean_waveforms[0].size())
    for i in range(len(clean_waveforms)):
        mean_mask += clean_waveforms[i] - noisy_waveforms[i]

    mean_mask /= len(clean_waveforms)

    # Apply the mask to the test noisy data
    for i in range(len(noisy_waveforms)):
        noisy_waveforms[i] += mean_mask

    # Save the waveforms
    # for i in range(len(noisy_waveforms)):
    #     torchaudio.save('data/test_baseline/' + noisy_filenames[i], noisy_waveforms[i], 16000)

    # Turn into torch tensors
    clean_waveforms = torch.stack(clean_waveforms)
    noisy_waveforms = torch.stack(noisy_waveforms)

    # Compute mean SNR
    snr = ScaleInvariantSignalNoiseRatio()
    snr_val = snr(clean_waveforms, noisy_waveforms)
    print('Mean SNR:', snr_val)

    # Compute mean eSTOI
    estoi = ShortTimeObjectiveIntelligibility(16000, extended=True)
    estoi_val = estoi(clean_waveforms, noisy_waveforms)
    print('Mean eSTOI:', estoi_val)

    # Compute mean SegSNR
    segsnr = SegSNR(seg_length=160)
    segsnr.update(clean_waveforms, noisy_waveforms)
    segsnr_val = segsnr.compute()
    print('Mean SegSNR:', segsnr_val)

    # Compute mean DNSMOS
    dnsmos_val = []
    for i in range(len(clean_waveforms)):
        dnsmos_val.append(dnsmos.run(noisy_waveforms[i].numpy(), 16000)['ovrl_mos'])
    print('Mean DNSMOS:', sum(dnsmos_val) / len(dnsmos_val))

    # Compute mean PESQ
    pesq = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
    pesq_val = pesq(clean_waveforms, noisy_waveforms)
    print('Mean PESQ:', pesq_val)


    
if __name__ == '__main__':
    baseline_model()

    # SNR: 8.3789
    # eSTOI: 0.6115
    # SegSNR: 4.7741
    # DNSMOS: 2.3418
