import matplotlib.pyplot as plt

# # Sample data: fractions of data used and their corresponding performance metrics.
# # Replace these with your actual fractions and performance metrics.
# total_training_examples = 11572
# fractions_of_data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
# training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# # Validation
# val_sisnr_10, val_sisnr_se_10 = 15.441953917440859, 0.14363883202515607
# val_sisnr_20, val_sisnr_se_20 = 16.47603084566524, 0.14294986711534546
# val_sisnr_30, val_sisnr_se_30 = 16.20867671636702, 0.15588626642327436
# val_sisnr_40, val_sisnr_se_40 = 16.483256264219005, 0.14216727486513273
# val_sisnr_50, val_sisnr_se_50 = 16.35517001991133, 0.14348859858726945
# val_sisnr_60, val_sisnr_se_60 = 16.175808554424822, 0.14096763352884184
# val_sisnr_70, val_sisnr_se_70 = 16.666244394570878, 0.1432565742760529
# val_sisnr_80, val_sisnr_se_80 = 16.648266483857793, 0.13842938219873466
# val_sisnr_90, val_sisnr_se_90 = 16.350830761553013, 0.1484269456651788
# val_sisnr_100, val_sisnr_se_100 = 16.107284186710828, 0.13237278740425423

# val_squim_mos_10, val_squim_mos_se_10 = 3.8459095251791684, 0.022879437318376196
# val_squim_mos_20, val_squim_mos_se_20 = 3.971659612597771, 0.02143530861088559
# val_squim_mos_30, val_squim_mos_se_30 = 3.91085997513197, 0.022633777779394293
# val_squim_mos_40, val_squim_mos_se_40 = 3.9448364793675617, 0.021412188796201936
# val_squim_mos_50, val_squim_mos_se_50 = 4.013431800800619, 0.020381643001799708
# val_squim_mos_60, val_squim_mos_se_60 = 3.997125227185129, 0.020627241630852583
# val_squim_mos_70, val_squim_mos_se_70 = 4.004416220685811, 0.020308686152108468
# val_squim_mos_80, val_squim_mos_se_80 = 4.090607931023663, 0.017952077167521528
# val_squim_mos_90, val_squim_mos_se_90 = 3.97871040748161, 0.020787025400390822
# val_squim_mos_100, val_squim_mos_se_100 = 4.046145101774086, 0.018766099495034026

# validation_sisnr = [val_sisnr_10, val_sisnr_20, val_sisnr_30, val_sisnr_40, val_sisnr_50, val_sisnr_60, val_sisnr_70, val_sisnr_80, val_sisnr_90, val_sisnr_100]
# validation_sisnr_se = [val_sisnr_se_10, val_sisnr_se_20, val_sisnr_se_30, val_sisnr_se_40, val_sisnr_se_50, val_sisnr_se_60, val_sisnr_se_70, val_sisnr_se_80, val_sisnr_se_90, val_sisnr_se_100]
# validation_squim_mos = [val_squim_mos_10, val_squim_mos_20, val_squim_mos_30, val_squim_mos_40, val_squim_mos_50, val_squim_mos_60, val_squim_mos_70, val_squim_mos_80, val_squim_mos_90, val_squim_mos_100]
# validation_squim_mos_se = [val_squim_mos_se_10, val_squim_mos_se_20, val_squim_mos_se_30, val_squim_mos_se_40, val_squim_mos_se_50, val_squim_mos_se_60, val_squim_mos_se_70, val_squim_mos_se_80, val_squim_mos_se_90, val_squim_mos_se_100]


# # Training
# train_sisnr_10, train_sisnr_se_10 = 11.751247124202413, 0.04897517145027679
# train_sisnr_20, train_sisnr_se_20 = 12.21057095263725, 0.049083701321878566
# train_sisnr_30, train_sisnr_se_30 = 12.043808353015011, 0.04934631147655701
# train_sisnr_40, train_sisnr_se_40 = 12.205102008810886, 0.048473134398917726
# train_sisnr_50, train_sisnr_se_50 = 12.24662312886478, 0.048285924858393885
# train_sisnr_60, train_sisnr_se_60 = 12.153604656483393, 0.047595365486127306
# train_sisnr_70, train_sisnr_se_70 = 12.438217402515495, 0.048380003854892356
# train_sisnr_80, train_sisnr_se_80 = 12.519000597700066, 0.04708759275166661
# train_sisnr_90, train_sisnr_se_90 = 12.347745827596533, 0.04796324465811897
# train_sisnr_100, train_sisnr_se_100 = 12.05933319001014, 0.04609703623147264

# train_squim_mos_10, train_squim_mos_se_10 = 3.44692169766998, 0.007685242973447401
# train_squim_mos_20, train_squim_mos_se_20 = 3.6065924139200836, 0.007327884343945209
# train_squim_mos_30, train_squim_mos_se_30 = 3.5381764278995345, 0.007461553508165439
# train_squim_mos_40, train_squim_mos_se_40 = 3.6469974181168308, 0.0072017856847729185
# train_squim_mos_50, train_squim_mos_se_50 = 3.6810038726592698, 0.007138933526150617
# train_squim_mos_60, train_squim_mos_se_60 = 3.6593500163536192, 0.007253502529366114
# train_squim_mos_70, train_squim_mos_se_70 = 3.6811951151200737, 0.007143417199912405
# train_squim_mos_80, train_squim_mos_se_80 = 3.7932277275664643, 0.006721966464551902
# train_squim_mos_90, train_squim_mos_se_90 = 3.7057521190974607, 0.007060354797931947
# train_squim_mos_100, train_squim_mos_se_100 = 3.7749090338923548, 0.006743689625758027


# training_sisnr = [train_sisnr_10, train_sisnr_20, train_sisnr_30, train_sisnr_40, train_sisnr_50, train_sisnr_60, train_sisnr_70, train_sisnr_80, train_sisnr_90, train_sisnr_100]
# training_sisnr_se = [train_sisnr_se_10, train_sisnr_se_20, train_sisnr_se_30, train_sisnr_se_40, train_sisnr_se_50, train_sisnr_se_60, train_sisnr_se_70, train_sisnr_se_80, train_sisnr_se_90, train_sisnr_se_100]
# training_squim_mos = [train_squim_mos_10, train_squim_mos_20, train_squim_mos_30, train_squim_mos_40, train_squim_mos_50, train_squim_mos_60, train_squim_mos_70, train_squim_mos_80, train_squim_mos_90, train_squim_mos_100]
# training_squim_mos_se = [train_squim_mos_se_10, train_squim_mos_se_20, train_squim_mos_se_30, train_squim_mos_se_40, train_squim_mos_se_50, train_squim_mos_se_60, train_squim_mos_se_70, train_squim_mos_se_80, train_squim_mos_se_90, train_squim_mos_se_100]





total_training_examples = 11572
fractions_of_data = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.]
fractions_of_data = [x*100 for x in fractions_of_data]
training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# Validation
val_sisnr_1, val_sisnr_se_1 = 12.539, 0
val_sisnr_2, val_sisnr_se_2 = 14.055, 0
val_sisnr_3, val_sisnr_se_3 = 14.736, 0
val_sisnr_5, val_sisnr_se_5 = 15.297, 0
val_sisnr_8, val_sisnr_se_8 = 16.006, 0
val_sisnr_10, val_sisnr_se_10 = 15.441953917440859, 0 #.14363883202515607
val_sisnr_20, val_sisnr_se_20 = 16.47603084566524, 0 #.14294986711534546
val_sisnr_30, val_sisnr_se_30 = 16.20867671636702, 0 #.15588626642327436
val_sisnr_40, val_sisnr_se_40 = 16.483256264219005, 0 #.14216727486513273
val_sisnr_50, val_sisnr_se_50 = 16.35517001991133, 0 #.14348859858726945
val_sisnr_60, val_sisnr_se_60 = 16.175808554424822, 0 #.14096763352884184
val_sisnr_70, val_sisnr_se_70 = 16.666244394570878, 0 #.1432565742760529
val_sisnr_80, val_sisnr_se_80 = 16.648266483857793, 0 #.13842938219873466
val_sisnr_90, val_sisnr_se_90 = 16.350830761553013, 0 #.1484269456651788
val_sisnr_100, val_sisnr_se_100 = 16.107284186710828, 0 #.13237278740425423

val_squim_mos_1, val_squim_mos_se_1 = 3.663, 0
val_squim_mos_2, val_squim_mos_se_2 = 3.705, 0
val_squim_mos_3, val_squim_mos_se_3 = 3.736, 0
val_squim_mos_5, val_squim_mos_se_5 = 3.91, 0
val_squim_mos_8, val_squim_mos_se_8 = 3.831, 0
val_squim_mos_10, val_squim_mos_se_10 = 3.8459095251791684, 0 #.022879437318376196
val_squim_mos_20, val_squim_mos_se_20 = 3.971659612597771, 0 #.02143530861088559
val_squim_mos_30, val_squim_mos_se_30 = 3.91085997513197, 0 #.022633777779394293
val_squim_mos_40, val_squim_mos_se_40 = 3.9448364793675617, 0 #.021412188796201936
val_squim_mos_50, val_squim_mos_se_50 = 4.013431800800619, 0 #.020381643001799708
val_squim_mos_60, val_squim_mos_se_60 = 3.997125227185129, 0 #.020627241630852583
val_squim_mos_70, val_squim_mos_se_70 = 4.004416220685811, 0 #.020308686152108468
val_squim_mos_80, val_squim_mos_se_80 = 4.090607931023663, 0 #.017952077167521528
val_squim_mos_90, val_squim_mos_se_90 = 3.97871040748161, 0 #.020787025400390822
val_squim_mos_100, val_squim_mos_se_100 = 4.046145101774086, 0 #.018766099495034026


validation_sisnr = [val_sisnr_1, val_sisnr_2, val_sisnr_3, val_sisnr_5, val_sisnr_8, val_sisnr_10, val_sisnr_20, val_sisnr_30, val_sisnr_40, val_sisnr_50, val_sisnr_60, val_sisnr_70, val_sisnr_80, val_sisnr_90, val_sisnr_100]
validation_sisnr_se = [val_sisnr_se_1, val_sisnr_se_2, val_sisnr_se_3, val_sisnr_se_5, val_sisnr_se_8, val_sisnr_se_10, val_sisnr_se_20, val_sisnr_se_30, val_sisnr_se_40, val_sisnr_se_50, val_sisnr_se_60, val_sisnr_se_70, val_sisnr_se_80, val_sisnr_se_90, val_sisnr_se_100]
validation_squim_mos = [val_squim_mos_1, val_squim_mos_2, val_squim_mos_3, val_squim_mos_5, val_squim_mos_8, val_squim_mos_10, val_squim_mos_20, val_squim_mos_30, val_squim_mos_40, val_squim_mos_50, val_squim_mos_60, val_squim_mos_70, val_squim_mos_80, val_squim_mos_90, val_squim_mos_100]
validation_squim_mos_se = [val_squim_mos_se_1, val_squim_mos_se_2, val_squim_mos_se_3, val_squim_mos_se_5, val_squim_mos_se_8, val_squim_mos_se_10, val_squim_mos_se_20, val_squim_mos_se_30, val_squim_mos_se_40, val_squim_mos_se_50, val_squim_mos_se_60, val_squim_mos_se_70, val_squim_mos_se_80, val_squim_mos_se_90, val_squim_mos_se_100]


ticks = [1, 3, 5, 8, 12, 15, 25, 40-5, 50-5, 60-5, 70-5, 80-5, 90-5, 100-5, 110-5]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20*0.7, 6*0.7))

# SI-SNR subplot
axs[0].plot(ticks, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
axs[0].set_title('Data Fraction Learning Curve (SI-SNR)')
axs[0].set_xlabel('Percentage of data used for training')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(ticks)  # Manually specify the ticks
axs[0].set_xticklabels(['1', '2', '3', '5', '8', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
axs[0].grid(axis='x')

# Squim MOS subplot
axs[1].plot(ticks, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
axs[1].set_title('Data Fraction Learning Curve (Squim MOS)')
axs[1].set_xlabel('Percentage of data used for training')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(ticks)  # Manually specify the ticks
axs[1].set_xticklabels(['1', '2', '3', '5', '8', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
axs[1].grid(axis='x')

plt.tight_layout()
plt.savefig('learning_curves_fraction.png', dpi=300)
# plt.show()

