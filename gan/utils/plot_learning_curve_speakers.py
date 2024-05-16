import matplotlib.pyplot as plt

fractions_of_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 28]

# Validation (epoch 499)
val_sisnr_1, val_sisnr_se_1 = 14.04, 0
val_sisnr_2, val_sisnr_se_2 = 12.94, 0
val_sisnr_3, val_sisnr_se_3 = 13.84, 0
val_sisnr_4, val_sisnr_se_4 = 16.41, 0
val_sisnr_5, val_sisnr_se_5 = 15.40, 0
val_sisnr_6, val_sisnr_se_6 = 16.02, 0
val_sisnr_7, val_sisnr_se_7 = 15.53, 0
val_sisnr_8, val_sisnr_se_8 = 15.48, 0
val_sisnr_9, val_sisnr_se_9 = 15.38, 0
val_sisnr_10, val_sisnr_se_10 = 14.95, 0
val_sisnr_15, val_sisnr_se_15 = 16.56, 0
val_sisnr_20, val_sisnr_se_20 = 16.58, 0 #491e
val_sisnr_25, val_sisnr_se_25 = 16.78, 0
val_sisnr_28, val_sisnr_se_28 = 16.107284186710828, 0# 0.13237278740425423

val_squim_mos_1, val_squim_mos_se_1 = 3.69, 0
val_squim_mos_2, val_squim_mos_se_2 = 3.86, 0
val_squim_mos_3, val_squim_mos_se_3 = 4.02, 0
val_squim_mos_4, val_squim_mos_se_4 = 3.90, 0
val_squim_mos_5, val_squim_mos_se_5 = 3.87, 0
val_squim_mos_6, val_squim_mos_se_6 = 3.93, 0
val_squim_mos_7, val_squim_mos_se_7 = 3.92, 0
val_squim_mos_8, val_squim_mos_se_8 = 3.95, 0
val_squim_mos_9, val_squim_mos_se_9 = 3.83, 0
val_squim_mos_10, val_squim_mos_se_10 = 4.00, 0
val_squim_mos_15, val_squim_mos_se_15 = 4.01, 0
val_squim_mos_20, val_squim_mos_se_20 = 3.90, 0
val_squim_mos_25, val_squim_mos_se_25 = 3.95, 0
val_squim_mos_28, val_squim_mos_se_28 =  4.046145101774086, 0# 0.018766099495034026

validation_sisnr = [val_sisnr_1, val_sisnr_2, val_sisnr_3, val_sisnr_4, val_sisnr_5, val_sisnr_6, val_sisnr_7, val_sisnr_8, val_sisnr_9, val_sisnr_10, val_sisnr_15, val_sisnr_20, val_sisnr_25, val_sisnr_28]
validation_sisnr_se = [val_sisnr_se_1, val_sisnr_se_2, val_sisnr_se_3, val_sisnr_se_4, val_sisnr_se_5, val_sisnr_se_6, val_sisnr_se_7, val_sisnr_se_8, val_sisnr_se_9, val_sisnr_se_10, val_sisnr_se_15, val_sisnr_se_20, val_sisnr_se_25, val_sisnr_se_28]
validation_squim_mos = [val_squim_mos_1, val_squim_mos_2, val_squim_mos_3, val_squim_mos_4, val_squim_mos_5, val_squim_mos_6, val_squim_mos_7, val_squim_mos_8, val_squim_mos_9, val_squim_mos_10, val_squim_mos_15, val_squim_mos_20, val_squim_mos_25, val_squim_mos_28]
validation_squim_mos_se = [val_squim_mos_se_1, val_squim_mos_se_2, val_squim_mos_se_3, val_squim_mos_se_4, val_squim_mos_se_5, val_squim_mos_se_6, val_squim_mos_se_7, val_squim_mos_se_8, val_squim_mos_se_9, val_squim_mos_se_10, val_squim_mos_se_15, val_squim_mos_se_20, val_squim_mos_se_25, val_squim_mos_se_28]


# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# SI-SNR subplot
# axs[0].plot(fractions_of_data, training_sisnr, label='Training SI-SNR', marker='o', color='blue')
# axs[0].fill_between(fractions_of_data, 
#                     [a - b for a, b in zip(training_sisnr, training_sisnr_se)], 
#                     [a + b for a, b in zip(training_sisnr, training_sisnr_se)], 
#                     color='blue', alpha=0.2)
axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
axs[0].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(validation_sisnr, validation_sisnr_se)], 
                    [a + b for a, b in zip(validation_sisnr, validation_sisnr_se)], 
                    color='red', alpha=0.2)
axs[0].set_title('SI-SNR Learning Curves')
axs[0].set_xlabel('Number of speakers')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(fractions_of_data)
axs[0].legend()

# Squim MOS subplot
# axs[1].plot(fractions_of_data, training_squim_mos, label='Training Squim MOS', marker='o',color='blue')
# axs[1].fill_between(fractions_of_data, 
#                     [a - b for a, b in zip(training_squim_mos, training_squim_mos_se)], 
#                     [a + b for a, b in zip(training_squim_mos, training_squim_mos_se)], 
#                     color='blue', alpha=0.2)
axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
axs[1].fill_between(fractions_of_data, 
                    [a - b for a, b in zip(validation_squim_mos, validation_squim_mos_se)], 
                    [a + b for a, b in zip(validation_squim_mos, validation_squim_mos_se)], 
                    color='red', alpha=0.2)
axs[1].set_title('Squim MOS Learning Curves')
axs[1].set_xlabel('Number of speakers')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(fractions_of_data)
axs[1].legend()

plt.savefig('learning_curves.png', dpi=300)
plt.show()


