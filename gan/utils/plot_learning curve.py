import matplotlib.pyplot as plt

# Sample data: fractions of data used and their corresponding performance metrics.
# Replace these with your actual fractions and performance metrics.
total_training_examples = 23252
fractions_of_data = [0.1, 0.2, 0.5, 0.7, 1.0] 
training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# Example performance metrics for training
training_sisnr = [12, 13, 14, 15, 16]
training_squim_mos = [2.9, 3.2, 3.6, 3.7, 3.8]

# Example performance metrics for validation
validation_sisnr = [11, 12, 13, 14, 15]
validation_squim_mos = [2.5, 2.9, 3.2, 3.6, 3.7]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# SI-SNR subplot
axs[0].plot(fractions_of_data, training_sisnr, label='Training SI-SNR', marker='o', color='blue')
axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
axs[0].set_title('SI-SNR Learning Curves')
axs[0].set_xlabel('Fraction of Data Used')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(fractions_of_data)
axs[0].legend()

# Squim MOS subplot
axs[1].plot(fractions_of_data, training_squim_mos, label='Training Squim MOS', marker='o', linestyle='--', color='blue')
axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', linestyle='--', color='red')
axs[1].set_title('Squim MOS Learning Curves')
axs[1].set_xlabel('Fraction of Data Used')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(fractions_of_data)
axs[1].legend()

plt.show()




