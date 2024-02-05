from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import io  # For handling in-memory files
import hydra


def visualize_stft_spectrogram_to_wandb(stft_data):
    """
    Visualizes an STFT-transformed file as a spectrogram and returns the plot as an image object
    for logging to wandb.
    
    Parameters:
    - stft_data: np.ndarray with shape (2, Frequency bins, Frames). The first dimension contains
                 the real and imaginary parts of the STFT.
    
    Returns:
    - A BytesIO object containing the image of the plot.
    """
    if stft_data.shape[0] != 2:
        raise ValueError("stft_data should have a shape (2, Frequency bins, Frames)")
    
    complex_stft = stft_data[0] + 1j * stft_data[1]
    magnitude_spectrum = np.abs(complex_stft)
    
    # Create a bytes buffer for the image
    buf = io.BytesIO()
    
    # Create figure without displaying it
    plt.figure(figsize=(10, 6))
    plt.imshow(magnitude_spectrum, aspect='auto', origin='lower', 
               extent=[0, magnitude_spectrum.shape[1], 0, magnitude_spectrum.shape[0]])
    plt.colorbar(label='Magnitude')
    plt.xlabel('Time (Frames)')
    plt.ylabel('Frequency (Bins)')
    plt.title('Amplitude Spectrogram')
    
    # Save the plot to the buffer
    plt.savefig(buf, format='png')
    # Important: Close the plot to free memory
    plt.close()
    
    # Reset buffer's cursor to the beginning
    buf.seek(0)
    image = Image.open(buf)
    return image


# Example usage with wandb:
import wandb
@hydra.main(config_name="config.yaml", config_path="config")
def main(cfg):
    wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity)

    stft_data = np.random.rand(2, 100, 200)  # Example STFT data
    image = visualize_stft_spectrogram_to_wandb(stft_data)
    wandb.log({"spectrogram": wandb.Image(image)})

