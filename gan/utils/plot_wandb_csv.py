import pandas as pd
import matplotlib.pyplot as plt

filename = 'FT_1,3,5,12s'
title = 'SI-SNR on Test Set during Supervised Fine-tuning'
xlabel = 'Epoch'
ylabel = 'SI-SNR value'
cols = [22, 16, 10, 4]
names = ['1 Speaker', '3 Speakers', '5 Speakers', '12 Speakers']

# Load the CSV file once (outside the loop)
csv_file = f'/Users/fredmac/Documents/DTU-FredMac/{filename}.csv'
data = pd.read_csv(csv_file)

# Create a single figure for all plots
plt.figure(figsize=(4*1.6, 3*1.5))  # Adjust the figsize as needed

for col, name in zip(cols, names):
    # Extract values and plot
    values = data.iloc[:, col]
    plt.plot(values, label=name)  # Add a label for the legend

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(axis='y')
plt.ticklabel_format(style='plain')
plt.legend()  # Add a legend to distinguish the lines

plt.tight_layout()
plt.savefig(f'/Users/fredmac/Downloads/bachelor_project/{filename}.png', dpi=300) 
