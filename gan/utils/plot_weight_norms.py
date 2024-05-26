import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

filename = 'Discriminator_fc_layers1.bias_norm'
title = 'Disc Linear Layer  Weights'
xlabel = 'Epoch'
ylabel = 'Norm'

def concatenate_columns_by_index(file_path):
    col_indices = [16, 10, 4]
    # Load the CSV file into a DataFrame
    df = pd.read_csv(file_path)
    # Extract and concatenate the values from the specified columns
    concatenated_list = []
    for index in col_indices:
        concatenated_list.extend(df.iloc[:, index].tolist())
    concatenated_list = [val for val in concatenated_list if not np.isnan(val)]
    return concatenated_list

# Example usage:
file_path = f'/Users/fredmac/Documents/DTU-FredMac/{filename}.csv'
values = concatenate_columns_by_index(file_path)[:1001]
print(len(values))

# Plot the values
plt.figure(figsize=(4, 3))  # Adjust the figsize to change the size of the plot
plt.plot(values)
plt.title(title)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.grid(axis='y')

plt.ticklabel_format(style='plain') # Disable scientific notation
plt.tight_layout()  # Add this line to adjust the spacing and make sure labels are within the frame

# Set the thousand separator
# plt.gca().get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

plt.savefig(f'/Users/fredmac/Documents/DTU-FredMac/{filename}.png', dpi=300)  # Save the plot to a file
