import matplotlib.pyplot as plt

# Sample data: fractions of data used and their corresponding performance metrics.
# Replace these with your actual fractions and performance metrics.
total_training_examples = 11572
fractions_of_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# Validation
val_sisnr_10, val_sisnr_se_10 = 15.441953917440859, 0.14363883202515607
val_sisnr_20, val_sisnr_se_20 = 16.47603084566524, 0.14294986711534546
val_sisnr_30, val_sisnr_se_30 = 16.20867671636702, 0.15588626642327436
val_sisnr_40, val_sisnr_se_40 = 16.483256264219005, 0.14216727486513273
val_sisnr_50, val_sisnr_se_50 = 16.35517001991133, 0.14348859858726945
val_sisnr_60, val_sisnr_se_60 = 16.175808554424822, 0.14096763352884184
val_sisnr_70, val_sisnr_se_70 = 16.666244394570878, 0.1432565742760529
val_sisnr_80, val_sisnr_se_80 = 16.648266483857793, 0.13842938219873466
val_sisnr_90, val_sisnr_se_90 = 16.350830761553013, 0.1484269456651788
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


# Training
train_sisnr_10, train_sisnr_se_10 = 17, 1
train_sisnr_20, train_sisnr_se_20 = 18, 1
train_sisnr_30, train_sisnr_se_30 = 18, 1
train_sisnr_40, train_sisnr_se_40 = 18, 1
train_sisnr_50, train_sisnr_se_50 = 18, 1
train_sisnr_60, train_sisnr_se_60 = 18, 1
train_sisnr_70, train_sisnr_se_70 = 18, 1
train_sisnr_80, train_sisnr_se_80 = 18, 1
train_sisnr_90, train_sisnr_se_90 = 18, 1
train_sisnr_100, train_sisnr_se_100 = 18, 1

train_squim_mos_10, train_squim_mos_se_10 = 4, 1
train_squim_mos_20, train_squim_mos_se_20 = 4, 1
train_squim_mos_30, train_squim_mos_se_30 = 5, 1
train_squim_mos_40, train_squim_mos_se_40 = 5, 1
train_squim_mos_50, train_squim_mos_se_50 = 5, 1
train_squim_mos_60, train_squim_mos_se_60 = 4, 1
train_squim_mos_70, train_squim_mos_se_70 = 4, 1
train_squim_mos_80, train_squim_mos_se_80 = 5, 1
train_squim_mos_90, train_squim_mos_se_90 = 5, 1
train_squim_mos_100, train_squim_mos_se_100 = 5, 1


training_sisnr = [train_sisnr_10, train_sisnr_20, train_sisnr_30, train_sisnr_40, train_sisnr_50, train_sisnr_60, train_sisnr_70, train_sisnr_80, train_sisnr_90, train_sisnr_100]
training_sisnr_se = [train_sisnr_se_10, train_sisnr_se_20, train_sisnr_se_30, train_sisnr_se_40, train_sisnr_se_50, train_sisnr_se_60, train_sisnr_se_70, train_sisnr_se_80, train_sisnr_se_90, train_sisnr_se_100]
training_squim_mos = [train_squim_mos_10, train_squim_mos_20, train_squim_mos_30, train_squim_mos_40, train_squim_mos_50, train_squim_mos_60, train_squim_mos_70, train_squim_mos_80, train_squim_mos_90, train_squim_mos_100]
training_squim_mos_se = [train_squim_mos_se_10, train_squim_mos_se_20, train_squim_mos_se_30, train_squim_mos_se_40, train_squim_mos_se_50, train_squim_mos_se_60, train_squim_mos_se_70, train_squim_mos_se_80, train_squim_mos_se_90, train_squim_mos_se_100]



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


