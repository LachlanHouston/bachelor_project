import matplotlib.pyplot as plt
from matplotlib.lines import Line2D 

# Sample data: fractions of data used and their corresponding performance metrics.
fractions_of_data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 28]
fractions_of_data_transformed = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16]  # Custom transformed scale

# Validation
val_sisnr_0, val_sisnr_se_0 = 17.07, 0.15
val_sisnr_1, val_sisnr_se_1 = 17.55, 0.15
val_sisnr_2, val_sisnr_se_2 = 17.82, 0.15
val_sisnr_3, val_sisnr_se_3 = 17.64, 0.15
val_sisnr_4, val_sisnr_se_4 = 18.31, 0.15
val_sisnr_5, val_sisnr_se_5 = 18.24, 0.15
val_sisnr_6, val_sisnr_se_6 = 18.93, 0.15
val_sisnr_7, val_sisnr_se_7 = 18.91, 0.15
val_sisnr_8, val_sisnr_se_8 = 19.11, 0.15
val_sisnr_10, val_sisnr_se_10 = 19.23, 0.15
val_sisnr_12, val_sisnr_se_12 = 19.26, 0.15
val_sisnr_14, val_sisnr_se_14 = 19.32, 0.15
val_sisnr_28, val_sisnr_se_28 = 19.45, 0.15

val_squim_mos_0, val_squim_mos_se_0 = 4.04, 0.02
val_squim_mos_1, val_squim_mos_se_1 = 4.09, 0.02
val_squim_mos_2, val_squim_mos_se_2 = 4.08, 0.02
val_squim_mos_3, val_squim_mos_se_3 = 4.12, 0.02
val_squim_mos_4, val_squim_mos_se_4 = 4.14, 0.02
val_squim_mos_5, val_squim_mos_se_5 = 4.16, 0.02
val_squim_mos_6, val_squim_mos_se_6 = 4.23, 0.02
val_squim_mos_7, val_squim_mos_se_7 = 4.21, 0.02
val_squim_mos_8, val_squim_mos_se_8 = 4.25, 0.02
val_squim_mos_10, val_squim_mos_se_10 = 4.22, 0.02
val_squim_mos_12, val_squim_mos_se_12 = 4.21, 0.02
val_squim_mos_14, val_squim_mos_se_14 = 4.23, 0.02
val_squim_mos_28, val_squim_mos_se_28 = 4.24, 0.02

validation_sisnr = [val_sisnr_0, val_sisnr_1, val_sisnr_2, val_sisnr_3, val_sisnr_4, val_sisnr_5, val_sisnr_6, val_sisnr_7, val_sisnr_8, val_sisnr_10, val_sisnr_12, val_sisnr_14, val_sisnr_28]
validation_sisnr_se = [val_sisnr_se_0, val_sisnr_se_1, val_sisnr_se_2, val_sisnr_se_3, val_sisnr_se_4, val_sisnr_se_5, val_sisnr_se_6, val_sisnr_se_7, val_sisnr_se_8, val_sisnr_se_10, val_sisnr_se_12, val_sisnr_se_14, val_sisnr_se_28]
validation_squim_mos = [val_squim_mos_0, val_squim_mos_1, val_squim_mos_2, val_squim_mos_3, val_squim_mos_4, val_squim_mos_5, val_squim_mos_6, val_squim_mos_7, val_squim_mos_8, val_squim_mos_10, val_squim_mos_12, val_squim_mos_14, val_squim_mos_28]
validation_squim_mos_se = [val_squim_mos_se_0, val_squim_mos_se_1, val_squim_mos_se_2, val_squim_mos_se_3, val_squim_mos_se_4, val_squim_mos_se_5, val_squim_mos_se_6, val_squim_mos_se_7, val_squim_mos_se_8, val_squim_mos_se_10, val_squim_mos_se_12, val_squim_mos_se_14, val_squim_mos_se_28]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15*0.6 , 6*0.6 ))  # 1 row, 2 columns

# Helper function to plot with the first point in green and last point in blue
def plot_with_highlight(ax, x, y, y_se, label, color):
    ax.plot(x, y, marker='o', color=color)
    ax.fill_between(x, 
                    [a - b for a, b in zip(y, y_se)], 
                    [a + b for a, b in zip(y, y_se)], 
                    color=color, alpha=0.2)
    ax.scatter(x[0], y[0], color='green', zorder=5)  # Highlight first point in green
    ax.scatter(x[-1], y[-1], color='blue', zorder=5)  # Highlight last point in blue

# SI-SNR subplot
plot_with_highlight(axs[0], fractions_of_data_transformed, validation_sisnr, validation_sisnr_se, 'SI-SNR', 'red')
axs[0].set_title('SI-SNR')
axs[0].set_xlabel('No. of speakers used in supervised training')
axs[0].set_ylabel('SI-SNR')
axs[0].set_xticks(fractions_of_data_transformed)
axs[0].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, '(...) 28'])

# Squim MOS subplot
plot_with_highlight(axs[1], fractions_of_data_transformed, validation_squim_mos, validation_squim_mos_se, 'Squim MOS', 'red')
axs[1].set_title('Squim MOS')
axs[1].set_xlabel('No. of speakers used in supervised training')
axs[1].set_ylabel('Squim MOS')
axs[1].set_xticks(fractions_of_data_transformed)
axs[1].set_xticklabels([0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, '(...) 28'])

# Custom legends
custom_lines = [Line2D([0], [0], color='green', marker='o', markersize=10, label='Unsupervised'),
                Line2D([0], [0], color='red', marker='o', markersize=10, lw=2, label='Semi-Supervised'),
                Line2D([0], [0], color='blue', marker='o', markersize=10, label='Supervised')]

# Adding legends to each subplot
axs[0].legend(handles=custom_lines, loc='lower right')
axs[1].legend(handles=custom_lines, loc='lower right')
plt.tight_layout() 

plt.savefig('finetune.png', dpi=300)
# plt.show()
