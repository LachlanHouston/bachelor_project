import librosa
import numpy as np
import os
import tqdm
import csv
import matplotlib.pyplot as plt

def create_csv():
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

if __name__ == '__main__':
    plot_pitch()