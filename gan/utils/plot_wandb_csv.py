import pandas as pd
import matplotlib.pyplot as plt

filename = 'val_mos'
title = 'Squim MOS (Test)'
xlabel = 'Epoch'
ylabel = 'MOS value'




# Load values from the CSV file
csv_file = f'/Users/fredmac/Documents/DTU-FredMac/{filename}.csv'  # Replace with the path to your CSV file
data = pd.read_csv(csv_file)

# Extract the values from the CSV
values = data.iloc[:, 4][:1001]  # Assumes the values are in the first column

# Plot the values
plt.figure(figsize=(4, 4))  # Adjust the figsize to change the size of the plot
plt.plot(values)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(axis='y')

# Disable scientific notation
plt.ticklabel_format(style='plain')
plt.tight_layout()  # Add this line to adjust the spacing and make sure labels are within the frame

# Set the thousand separator
# plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig(f'/Users/fredmac/Documents/DTU-FredMac/{filename}.png')  # Save the plot to a file

