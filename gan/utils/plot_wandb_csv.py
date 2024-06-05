import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

filename = 'standard_squim'
title = 'Squim MOS on Test Set during Training'
xlabel = 'Epoch'
ylabel = 'MOS'

# Load the CSV file once (outside the loop)
csv_file = f'/Users/fredmac/Documents/DTU-FredMac/{filename}.csv'
data = pd.read_csv(csv_file)

col = 0
values = [x for x in np.array(data.iloc[:, col])]

# Create a single figure for all plots
plt.figure(figsize=(4*1.6, 3*1.5))  # Adjust the figsize as needed

plt.plot(values)  # Add a label for the legend

plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(axis='y')
plt.ticklabel_format(style='plain')
plt.legend()  # Add a legend to distinguish the lines

plt.tight_layout()
plt.savefig(f'/Users/fredmac/Downloads/bachelor_project/{filename}.png', dpi=300) 
