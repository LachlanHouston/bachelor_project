import pandas as pd
import matplotlib.pyplot as plt

filename = 'AS_pre_D_out'
title = 'Discriminator Outputs on AudioSet after Pretraining'
xlabel = 'Epoch'
ylabel = 'Output value'
cols = [7, 4]
names = ['Real', 'Fake']

# Load the CSV file once (outside the loop)
csv_file = f'/Users/fredmac/Documents/DTU-FredMac/{filename}.csv'
data = pd.read_csv(csv_file)

# Create a single figure for all plots
plt.figure(figsize=(4*1.3, 3*1.3))  # Adjust the figsize as needed

for col, name in zip(cols, names):
    # Extract values and plot
    values = data.iloc[:, col][:1001]
    plt.plot(values, label=name)  # Add a label for the legend

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(axis='y')
plt.ticklabel_format(style='plain')
plt.legend()  # Add a legend to distinguish the lines

plt.tight_layout()
plt.savefig(f'/Users/fredmac/Downloads/bachelor_project/{filename}.png', dpi=300) 
