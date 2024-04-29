import librosa
import numpy as np
import os
import tqdm
import csv
import matplotlib.pyplot as plt
import torchaudio
import torch

noisy_AudioSet_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_raw'
noisy_VCTKD_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_raw'
paths = [noisy_AudioSet_path, noisy_VCTKD_path]
csv_filenames = ['_AudioSet', '_VCTKD']
device = torch.device('cpu')

def create_csv_pitch():

    noisy_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/noisy_raw/'
    noisy_filenames = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

    with open('pitch_VCTK.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Pitch"])
        for filename in tqdm.tqdm(noisy_filenames):
            # Load an audio file
            y, sr = librosa.load(os.path.join(noisy_path, filename))

            # Compute the pitch (set fmin and fmax for voice frequency range)
            pitches, magnitudes = librosa.piptrack(y=y, sr=sr, fmin=75, fmax=300)

            # Selecting the dominant pitch per frame
            def extract_pitch(pitches, magnitudes):
                indices = magnitudes.argmax(axis=0)
                dominant_pitches = pitches[indices, range(pitches.shape[1])]
                return dominant_pitches

            dominant_pitches = extract_pitch(pitches, magnitudes)

            # You might want to filter out low amplitudes
            filtered_pitches = dominant_pitches[magnitudes.max(axis=0) > np.median(magnitudes.max(axis=0))]

            # Average pitch value
            average_pitch = np.mean(filtered_pitches)

            writer.writerow([filename, average_pitch])

def create_csv_zcr():
    noisy_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_raw'
    noisy_filenames = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

    with open('zcr_AudioSet.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Zero-Crossing Rate"])
        for filename in tqdm.tqdm(noisy_filenames):
            y, sr = librosa.load(os.path.join(noisy_path, filename))
            # Compute the zero-crossing rate
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            # Average pitch value
            average_zcr = np.mean(zcr)
            writer.writerow([filename, average_zcr])

def create_csv_timbre_composite():
    """
    Compute a simplified timbre composite for a given audio file.
    """
    noisy_path = '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/data/AudioSet/train_raw'
    noisy_filenames = sorted([file for file in os.listdir(noisy_path) if file.endswith('.wav')])

    with open('timbre_AudioSet.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Filename", "Timbre Composite"])
        for filename in tqdm.tqdm(noisy_filenames):
            # Load the audio file
            y, sr = librosa.load(os.path.join(noisy_path, filename))
            
            # Compute the Short-Time Fourier Transform (STFT)
            S = np.abs(librosa.stft(y))
            
            # Compute spectral features
            centroid = librosa.feature.spectral_centroid(S=S, sr=sr)
            contrast = librosa.feature.spectral_contrast(S=S, sr=sr)
            bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr)
            rolloff = librosa.feature.spectral_rolloff(S=S, sr=sr)
            
            # Normalize each feature to a 0-1 scale
            centroid_normalized = centroid / np.max(centroid)
            contrast_normalized = np.mean(contrast, axis=0) / np.max(np.mean(contrast, axis=0))
            bandwidth_normalized = bandwidth / np.max(bandwidth)
            rolloff_normalized = rolloff / np.max(rolloff)
            
            # Compute the average of the normalized features
            timbre_composite = np.mean([centroid_normalized.squeeze(), contrast_normalized, bandwidth_normalized.squeeze(), rolloff_normalized.squeeze()])
            writer.writerow([filename, timbre_composite])

def create_csvs_rms():
    for path, csv_filename in zip(paths, csv_filenames):
        noisy_filenames = sorted([file for file in os.listdir(path) if file.endswith('.wav')])

        with open(f'rms{csv_filename}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Filename", "Root Mean Square"])
            for filename in tqdm.tqdm(noisy_filenames):
                wav = torchaudio.load(os.path.join(path, filename))[0].to(device)
                # Compute the root mean square (RMS)
                rms = torch.sqrt(torch.mean(wav**2)).item()
                writer.writerow([filename, rms])

def create_csvs_mfcc():
    for path, csv_filename in zip(paths, csv_filenames):
        noisy_filenames = sorted([file for file in os.listdir(path) if file.endswith('.wav')])

        with open(f'mfcc{csv_filename}.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            # Adjust number of coefficients as needed; default here is 13
            
            writer.writerow(["Filename"] + [f"MFCC_{i}" for i in range(13)])
            for filename in tqdm.tqdm(noisy_filenames):
                wav, sample_rate = torchaudio.load(os.path.join(path, filename))
                mfcc_transform = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=2).to(device)
                wav = wav.to(device)
                # Compute the MFCCs
                mfcc = mfcc_transform(wav).mean(dim=2).squeeze().tolist()  # Mean across time frames
                writer.writerow([filename] + mfcc)


def plot_pitch_timbre():
    csvfile = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/scores/pitch/pitch_AudioSet.csv', 'r')
    reader = csv.reader(csvfile)
    next(reader)
    pitches = []
    for row in reader:
        pitches.append(float(row[1]))
    csvfile.close()  # Close the file after reading
    pitches_AS = np.array(pitches)

    csvfile2 = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/timbre_AudioSet.csv', 'r')
    reader2 = csv.reader(csvfile2)
    next(reader2)
    timbres = []
    for row in reader2:
        timbres.append(float(row[1]))
    csvfile2.close()  # Close the file after reading
    timbres_AS = np.array(timbres)

    points_AS = np.vstack((pitches_AS, timbres_AS)).T

    csvfile3 = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/scores/pitch/pitch_VCTK.csv', 'r')
    reader3 = csv.reader(csvfile3)
    next(reader3)
    pitches2 = []
    for row in reader3:
        pitches2.append(float(row[1]))
    csvfile3.close()  # Close the file after reading
    pitches_VCTK = np.array(pitches2)

    csvfile4 = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/timbre_VCTK.csv', 'r')
    reader4 = csv.reader(csvfile4)
    next(reader4)
    timbres2 = []
    for row in reader4:
        timbres2.append(float(row[1]))
    csvfile4.close()  # Close the file after reading
    timbres_VCTK = np.array(timbres2)

    points_VCTK = np.vstack((pitches_VCTK, timbres_VCTK)).T

    # Create scatter plots
    plt.scatter(points_AS[:, 0], points_AS[:, 1], alpha=0.2, color='red', label='AudioSet')
    plt.scatter(points_VCTK[:, 0], points_VCTK[:, 1], alpha=0.2, color='blue', label='VCTK')

    # Add legend, labels, title and show plot
    plt.legend(loc='upper right')
    plt.xlabel('Pitch (Hz)')
    plt.ylabel('Timbre Composite')
    plt.title('Pitch and Timbre Composite')
    plt.show()

def plot_pitch():
    csvfile = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/pitch_AudioSet.csv', 'r')
    reader = csv.reader(csvfile)
    next(reader)
    pitches = []
    for row in reader:
        pitches.append(float(row[1]))
    csvfile.close()  # Close the file after reading
    pitches = np.array(pitches)

    csvfile2 = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/pitch_VCTK.csv', 'r')
    reader2 = csv.reader(csvfile2)
    next(reader2)
    pitches2 = []
    for row in reader2:
        pitches2.append(float(row[1]))
    csvfile2.close()  # Close the file after reading
    pitches2 = np.array(pitches2)

    # Create histograms
    plt.hist(pitches, bins=50, weights=np.ones_like(pitches) / len(pitches), alpha=0.5, color='red', label='AudioSet')
    plt.hist(pitches2, bins=50, weights=np.ones_like(pitches2) / len(pitches2), alpha=0.5, color='blue', label='VCTKD')

    # Add legend, labels, title and show plot
    plt.legend(loc='upper right')
    plt.xlabel('Pitch (Hz)')
    plt.ylabel('Fraction of Data')
    plt.title('Pitch Distributions')
    plt.savefig('pitches.png', dpi=300)

def plot_zcr():
    plt.figure(figsize=(8, 6))

    csvfile = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/zcr_AudioSet.csv', 'r')
    reader = csv.reader(csvfile)
    next(reader)
    zcrs = []
    for row in reader:
        zcrs.append(float(row[1]))
    csvfile.close()  # Close the file after reading
    zcrs = np.array(zcrs)

    csvfile2 = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/zcr_VCTK.csv', 'r')
    reader2 = csv.reader(csvfile2)
    next(reader2)
    zcrs2 = []
    for row in reader2:
        zcrs2.append(float(row[1]))
    csvfile2.close()  # Close the file after reading
    zcrs2 = np.array(zcrs2)

    # Create histograms
    plt.hist(zcrs, bins=50, density=True, weights=np.ones_like(zcrs) / len(zcrs), alpha=0.5, color='red', label='AudioSet')
    plt.hist(zcrs2, bins=50, density=True, weights=np.ones_like(zcrs2) / len(zcrs2), alpha=0.5, color='blue', label='VCTKD')

    # Set the x-axis limit
    plt.xlim(0, 0.45)

    # Add legend, labels, title and show plot
    plt.legend(loc='upper right')
    plt.xlabel('Zero-Crossing Rate')
    plt.ylabel('Density')
    plt.title('Zero-Crossing Rate Distributions')
    plt.savefig('zcrs.png', dpi=300)

def plot_rms(csv_filenames, labels):
    # Set up the figure for plotting
    plt.figure(figsize=(8, 6))
    colors = ['red', 'blue']
    # Process each CSV file
    for csv_filename, label, color in zip(csv_filenames, labels, colors):
        # Open and read the CSV file
        with open(csv_filename, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header
            rms_values = [float(row[1]) for row in reader]  # Extract RMS values from each row

        # Convert list to numpy array for better handling
        rms_array = np.array(rms_values)

        # Plot the histogram of RMS values
        plt.hist(rms_array, bins=50, density=True, alpha=0.5, label=label, color=color)  # Remove '.csv' from label

    # Adding legends, labels and title to the plot
    plt.legend(title="Dataset")
    plt.xlim(0, 0.5)
    plt.xlabel('Root Mean Square Value')
    plt.ylabel('Density')
    plt.title('Root Mean Square Distributions')
    plt.savefig('rms_distribution.png', dpi=300)  # Save the plot as a PNG file
    # plt.show()

def plot_mfcc():
    plt.figure(figsize=(8, 6))

    # create scatterplots of the first two MFCCs
    csvAS = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/data_stats/mfcc_AudioSet.csv', 'r')
    readerAS = csv.reader(csvAS)
    next(readerAS)
    csvVCTKD = open('/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/data_stats/mfcc_VCTKD.csv', 'r')
    readerVCTKD = csv.reader(csvVCTKD)
    next(readerVCTKD)
    mfccsAS = []
    mfccsVCTKD = []
    for row in readerAS:
        mfccsAS.append([float(i) for i in row[1:]])
    for row in readerVCTKD:
        mfccsVCTKD.append([float(i) for i in row[1:]])
    csvAS.close()
    csvVCTKD.close()
    mfccsAS = np.array(mfccsAS)
    mfccsVCTKD = np.array(mfccsVCTKD)

    plt.scatter(mfccsAS[:, 0], mfccsAS[:, 1], alpha=0.2, color='red', label='AudioSet')
    plt.scatter(mfccsVCTKD[:, 0], mfccsVCTKD[:, 1], alpha=0.2, color='blue', label='VCTKD')
    plt.legend(loc='upper right')
    plt.xlabel('MFC Coefficient 1')
    plt.ylabel('MFC Coefficient 2')
    plt.title('Mel-Frequency Cepstral Coefficients')
    plt.savefig('mfccs.png', dpi=300)
    # plt.show()

if __name__ == '__main__':
    # create_csv_timbre_composite()
    # plot_pitch_timbre()
    # create_csv_zcr()
    # create_csvs_rms()
    # create_csvs_mfcc()
    # plot_zcr()
    # plot_pitch()
    plot_mfcc()

    # filenames = ['/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/data_stats/rms_AudioSet.csv', 
    #              '/Users/fredmac/Library/CloudStorage/OneDrive-DanmarksTekniskeUniversitet/bachelor_project/reports/data_stats/rms_VCTKD.csv']
    # labels = ['AudioSet', 'VCTKD']
    # plot_rms(filenames, labels)