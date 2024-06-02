import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = 'supervised_squim'
title = 'Squim MOS on Test Set during Training'
xlabel = 'Epoch'
ylabel = 'MOS'
cols = [22, 16, 10, 4]
names = ['1 Speaker', '3 Speakers', '5 Speakers', '12 Speakers']

# Load the CSV file once (outside the loop)
csv_file = '/Users/fredmac/Documents/DTU-FredMac/output, statistics/scores during training/squim_supervised.csv'
data = pd.read_csv(csv_file)
 
values = np.array([])
for col, name in zip(cols, names):
    # Extract values and plot
    print(col, len(data.iloc[:, col]))
    values = np.append(
        values, 
        [x for x in np.array(data.iloc[:, col]) if not np.isnan(x)]
        )
    print(len(values))

x_values = np.arange(len(values))
if 'squim' in filename:
    values = values[:101]
    x_values = x_values[:101]
    x_values *= 10
else:
    values = values[:1001]
    x_values = x_values[:1001]


# Create a single figure for all plots
plt.figure(figsize=(4*1.6, 3*1.5))  # Adjust the figsize as needed

# Generate the x-axis values, multiply by 10

plt.plot(x_values, values)  # Add a label for the legend

plt.title(title, fontsize=16)
plt.xlabel(xlabel, fontsize=14)
plt.ylabel(ylabel, fontsize=14)
plt.grid(axis='y')
plt.ticklabel_format(style='plain')
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()
plt.savefig(f'/Users/fredmac/Downloads/bachelor_project/{filename}.png', dpi=300) 
