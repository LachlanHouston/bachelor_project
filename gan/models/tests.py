import torch
import torch.nn as nn
waveform, sample_rate, spectrogram = torch.load("data/clean_processed/0.pt")

# Define the LSTM model
input_size = spectrogram.shape[1]  # Number of input features
hidden_size = 64  # Number of LSTM units
num_layers = 2  # Number of LSTM layers
batch_first = True  # Set to True to have batch as the first dimension
dropout = 0.2  # Dropout probability
lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first, dropout=dropout)

# Reshape the spectrogram to match the expected input shape of the LSTM
# spectrogram = spectrogram.unsqueeze(0)  # Add a batch dimension
print(spectrogram.shape)
# Pass the spectrogram through the LSTM
output, (h_n, c_n) = lstm(spectrogram)

# Print the shape of the LSTM output
print("LSTM output shape:", output.shape)