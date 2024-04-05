import torch
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio
from torchmetrics.audio import ShortTimeObjectiveIntelligibility
from torchaudio.pipelines import SQUIM_SUBJECTIVE, SQUIM_OBJECTIVE
from speechmos import dnsmos

def compute_scores(real_clean_waveform, fake_clean_waveform, non_matching_reference_waveform, use_pesq=True):

    if real_clean_waveform.numpy().shape == (1, 32000):
        real_clean_waveform = real_clean_waveform.squeeze(0)
    if fake_clean_waveform.numpy().shape == (1, 32000):
        fake_clean_waveform = fake_clean_waveform.squeeze(0)
    if len(non_matching_reference_waveform.numpy().shape) == 1:
        non_matching_reference_waveform = non_matching_reference_waveform.unsqueeze(0)

    ## SI-SNR
    sisnr = ScaleInvariantSignalNoiseRatio()
    sisnr_score = sisnr(preds=fake_clean_waveform, target=real_clean_waveform)

    ## DNSMOS
    dnsmos_score = dnsmos.run(fake_clean_waveform.numpy(), 16000)['ovrl_mos']

    ## MOS Squim
    subjective_model = SQUIM_SUBJECTIVE.get_model()
    mos_squim_score = subjective_model(fake_clean_waveform.unsqueeze(0), non_matching_reference_waveform)
        
    ## eSTOI
    estoi = ShortTimeObjectiveIntelligibility(16000, extended = True)
    estoi_score = estoi(preds = fake_clean_waveform, target = real_clean_waveform)

    if use_pesq:
        from torchmetrics.audio import PerceptualEvaluationSpeechQuality
        from pesq import pesq
        ## PESQ Normal
        pesq_normal_score = pesq(fs=16000, ref=real_clean_waveform.numpy(), deg=fake_clean_waveform.numpy(), mode='wb')

        ## PESQ Torch
        pesq_torch = PerceptualEvaluationSpeechQuality(fs=16000, mode='wb')
        pesq_torch_score = pesq_torch(real_clean_waveform, fake_clean_waveform)

    ## Predicted objective metrics: STOI, PESQ, and SI-SDR
    objective_model = SQUIM_OBJECTIVE.get_model()
    stoi_pred, pesq_pred, si_sdr_pred = objective_model(fake_clean_waveform.unsqueeze(0))
    if use_pesq:
        return sisnr_score.item(), dnsmos_score, mos_squim_score.item(), estoi_score.item(), pesq_normal_score, pesq_torch_score.item(), stoi_pred.item(), pesq_pred.item(), si_sdr_pred.item()
    return sisnr_score.item(), dnsmos_score, mos_squim_score.item(), estoi_score.item(), 0,                 0,                       stoi_pred.item(), pesq_pred.item(), si_sdr_pred.item()

def perfect_shuffle(tensor):
    # Ensure the tensor is at least 2D
    if tensor.dim() < 2:
        raise ValueError("Tensor must be at least 2-dimensional")

    size = tensor.size(0)
    idx = torch.randperm(size)
    
    # Check if the shuffle is a perfect derangement
    while (idx == torch.arange(size)).any():
        idx = torch.randperm(size)

    return tensor[idx]


def stft_to_waveform(stft, device=torch.device('cuda')):
    if len(stft.shape) == 3:
        stft = stft.unsqueeze(0)
    # Separate the real and imaginary components
    stft_real = stft[:, 0, :, :]
    stft_imag = stft[:, 1, :, :]
    # Combine the real and imaginary components to form the complex-valued spectrogram
    stft = torch.complex(stft_real, stft_imag)
    # Perform inverse STFT to obtain the waveform
    waveform = torch.istft(stft, n_fft=512, hop_length=100, win_length=400, window=torch.hann_window(400).to(device))
    return waveform


def visualize_stft_spectrogram(real_clean, fake_clean, real_noisy):
    """
    Visualizes a STFT-transformed files as mel spectrograms and returns the plot as an image object
    for logging to wandb.
    """    

    S_real_c = real_clean[0].cpu()
    S_fake_c = fake_clean[0].cpu()
    S_real_n = real_noisy[0].cpu()

    # Spectrogram of real clean
    mel_spect_rc = librosa.feature.melspectrogram(S=S_real_c, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_rc = librosa.power_to_db(mel_spect_rc, ref=np.max)
    # Spectrogram of fake clean
    mel_spect_fc = librosa.feature.melspectrogram(S=S_fake_c, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_fc = librosa.power_to_db(mel_spect_fc, ref=np.max)
    # Spectrogram of real noisy
    mel_spect_rn = librosa.feature.melspectrogram(S=S_real_n, sr=16000, n_fft=512, hop_length=100, power=2)
    mel_spect_db_rn = librosa.power_to_db(mel_spect_rn, ref=np.max)
    
    # Create a figure with 3 subplots
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Define Real Clean plot
    img_rc = librosa.display.specshow(mel_spect_db_rc, ax=axs[0], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_rc, ax=axs[0], format='%+2.0f dB')
    axs[0].set_title('Real Clean')
    axs[0].set_xlabel('Time (s)')
    axs[0].set_ylabel('Frequency (Hz)')

    # Define Fake Clean plot
    img_fc = librosa.display.specshow(mel_spect_db_fc, ax=axs[1], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_fc, ax=axs[1], format='%+2.0f dB')
    axs[1].set_title('Fake Clean')
    axs[1].set_xlabel('Time (s)')
    axs[1].set_ylabel('Frequency (Hz)')

    # Define Real Noisy plot
    img_rn = librosa.display.specshow(mel_spect_db_rn, ax=axs[2], y_axis='mel', fmax=8000, x_axis='time', hop_length=100, sr=16000)
    fig.colorbar(img_rn, ax=axs[2], format='%+2.0f dB')
    axs[2].set_title('Real Noisy')
    axs[2].set_xlabel('Time (s)')
    axs[2].set_ylabel('Frequency (Hz)')

    # Set the title of the figure
    fig.suptitle('Spectrograms')
    plt.tight_layout(pad=3.0)
    
    return plt
