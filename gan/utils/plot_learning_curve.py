import matplotlib.pyplot as plt

# Sample data: fractions of data used and their corresponding performance metrics.
# Replace these with your actual fractions and performance metrics.
total_training_examples = 11572
fractions_of_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# Training
training_sisnr = [12, 13, 14, 15, 16,16, 17, 18, 19, 20]
training_sisnr_se = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
training_squim_mos = [2.9, 3.2, 3.6, 3.7, 3.8, 2.9, 3.2, 3.6, 3.7, 3.8]
training_squim_mos_se = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

# Validation
val_sisnr_10, val_sisnr_se_10 = 11, 0.1
val_sisnr_20, val_sisnr_se_20 = 12, 0.1
val_sisnr_30, val_sisnr_se_30 = 13, 0.1
val_sisnr_40, val_sisnr_se_40 = 14, 0.1
val_sisnr_50, val_sisnr_se_50 = 15, 0.1
val_sisnr_60, val_sisnr_se_60 = 16, 0.1
val_sisnr_70, val_sisnr_se_70 = 12, 0.1
val_sisnr_80, val_sisnr_se_80 = 13, 0.1
val_sisnr_90, val_sisnr_se_90 = 14, 0.1
val_sisnr_100, val_sisnr_se_100 = 16.107284186710828, 0.13237278740425423

val_squim_mos_10, val_squim_mos_se_10 = 2.5, 0.1
val_squim_mos_20, val_squim_mos_se_20 = 2.9, 0.1
val_squim_mos_30, val_squim_mos_se_30 = 3.2, 0.1
val_squim_mos_40, val_squim_mos_se_40 = 3.6, 0.1
val_squim_mos_50, val_squim_mos_se_50 = 3.7, 0.1
val_squim_mos_60, val_squim_mos_se_60 = 2.5, 0.1
val_squim_mos_70, val_squim_mos_se_70 = 2.9, 0.1
val_squim_mos_80, val_squim_mos_se_80 = 3.2, 0.1
val_squim_mos_90, val_squim_mos_se_90 = 3.6, 0.1
val_squim_mos_100, val_squim_mos_se_100 = 3.7, 0.1

validation_sisnr = [val_sisnr_10, val_sisnr_20, val_sisnr_30, val_sisnr_40, val_sisnr_50, val_sisnr_60, val_sisnr_70, val_sisnr_80, val_sisnr_90, val_sisnr_100]
validation_sisnr_se = [val_sisnr_se_10, val_sisnr_se_20, val_sisnr_se_30, val_sisnr_se_40, val_sisnr_se_50, val_sisnr_se_60, val_sisnr_se_70, val_sisnr_se_80, val_sisnr_se_90, val_sisnr_se_100]
validation_squim_mos = [val_squim_mos_10, val_squim_mos_20, val_squim_mos_30, val_squim_mos_40, val_squim_mos_50, val_squim_mos_60, val_squim_mos_70, val_squim_mos_80, val_squim_mos_90, val_squim_mos_100]
validation_squim_mos_se = [val_squim_mos_se_10, val_squim_mos_se_20, val_squim_mos_se_30, val_squim_mos_se_40, val_squim_mos_se_50, val_squim_mos_se_60, val_squim_mos_se_70, val_squim_mos_se_80, val_squim_mos_se_90, val_squim_mos_se_100]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# SI-SNR subplot
axs[0].plot(fractions_of_data, training_sisnr, label='Training SI-SNR', marker='o', color='blue')
axs[0].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(training_sisnr, training_sisnr_se)], 
                    [a + b for a, b in zip(training_sisnr, training_sisnr_se)], 
                    color='blue', alpha=0.2)
axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
axs[0].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(validation_sisnr, validation_sisnr_se)], 
                    [a + b for a, b in zip(validation_sisnr, validation_sisnr_se)], 
                    color='red', alpha=0.2)
axs[0].set_title('SI-SNR Learning Curves')
axs[0].set_xlabel('Fraction of Data Used')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(fractions_of_data)
axs[0].legend()

# Squim MOS subplot
axs[1].plot(fractions_of_data, training_squim_mos, label='Training Squim MOS', marker='o',color='blue')
axs[1].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(training_squim_mos, training_squim_mos_se)], 
                    [a + b for a, b in zip(training_squim_mos, training_squim_mos_se)], 
                    color='blue', alpha=0.2)
axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
axs[1].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(validation_squim_mos, validation_squim_mos_se)], 
                    [a + b for a, b in zip(validation_squim_mos, validation_squim_mos_se)], 
                    color='red', alpha=0.2)
axs[1].set_title('Squim MOS Learning Curves')
axs[1].set_xlabel('Fraction of Data Used')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(fractions_of_data)
axs[1].legend()

plt.show()



