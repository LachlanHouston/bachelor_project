import matplotlib.pyplot as plt

fractions_of_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 28]

# Initialize lists for storing SI-SNR and Squim MOS values
# Lists are formatted as: [Ordering 1, Ordering 2, Ordering 3]
val_sisnr = [
    [0, 0, 0],  # 1 Speaker
    [0, 0, 0],  # 2 Speakers
    [0, 0, 0],  # 3 Speakers
    [0, 0, 0],  # 4 Speakers
    [0, 0, 0],  # 5 Speakers
    [0, 0, 0],  # 6 Speakers
    [0, 17.26300370519600, 0],  # 7 Speakers
    [0, 0, 0],  # 8 Speakers
    [0, 0, 0],  # 9 Speakers
    [0, 0, 0],  # 10 Speakers
    [0, 0, 0],  # 15 Speakers
    [0, 0, 0],  # 20 Speakers
    [0, 0, 0],  # 25 Speakers
    [0, 16.742823031342100, 0]   # 28 Speakers
]

val_squim_mos = [
    [0, 0, 0],  # 1 Speaker
    [0, 0, 0],  # 2 Speakers
    [0, 0, 0],  # 3 Speakers
    [0, 0, 0],  # 4 Speakers
    [0, 0, 0],  # 5 Speakers
    [0, 0, 0],  # 6 Speakers
    [0, 4.026432449956540, 0],  # 7 Speakers
    [0, 0, 0],  # 8 Speakers
    [0, 0, 0],  # 9 Speakers
    [0, 0, 0],  # 10 Speakers
    [0, 0, 0],  # 15 Speakers
    [0, 0, 0],  # 20 Speakers
    [0, 0, 0],  # 25 Speakers
    [0, 0, 0]   # 28 Speakers
]


# Calculate means
validation_sisnr = [sum(vals) / len(vals) for vals in val_sisnr]
validation_squim_mos = [sum(vals) / len(vals) for vals in val_squim_mos]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# SI-SNR subplot
axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
axs[0].set_title('Speaker-separated Learning Curve (SI-SNR)')
axs[0].set_xlabel('Number of speakers')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(fractions_of_data)
axs[0].legend()

# Squim MOS subplot
axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
axs[1].set_title('Speaker-separated Learning Curve (Squim MOS)')
axs[1].set_xlabel('Number of speakers')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(fractions_of_data)
axs[1].legend()

plt.savefig('learning_curves.png', dpi=300)
plt.show()
