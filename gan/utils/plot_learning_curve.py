import matplotlib.pyplot as plt

total_training_examples = 11572
fractions_of_data = [0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
percentages_of_data = [x * 100 for x in fractions_of_data]
training_examples = [int(total_training_examples * frac) for frac in fractions_of_data]

# Combined validation SI-SNR and Squim MOS values
validation_data = [
    [13.733627011764396 , 3.4457068486699782],  # 1% of data
    [15.210412041076179 , 3.5583383079292705],  # 2% of data
    [ 14.978912165153400, 3.5862344986605400],  # 3% of data
    [ 16.018059443501600, 3.690953813710260],  # 5% of data
    [ 16.56694811351090,3.8811491343002900 ],  # 8% of data
    [16.837263298555500 ,3.896116212444400 ],  # 10% of data
    [16.932354658697400 , 3.9308188985852400],  # 20% of data
    [ 16.700575750719000, 3.9151245592867300],  # 30% of data
    [16.611505038819300 , 3.93997752724342],  # 40% of data
    [ 16.599455413887800, 3.967496260855960],  # 50% of data
    [ 16.81806779197120, 3.9634332428279400],  # 60% of data
    [ 16.865651221819300, 4.005131580007890],  # 70% of data
    [ 16.80847135244060, 3.941426294810560],  # 80% of data
    [16.78949373381810 , 3.9953767492354500],  # 90% of data
    [16.78933193498450,4.072483748776240 ]   # 100% of data
]


ticks = [1, 3, 5, 8, 12, 15, 25, 35, 45, 55, 65, 75, 85, 95, 105]

# Creating subplots
fig, axs = plt.subplots(1, 2, figsize=(20*0.6, 6*0.6 ))

# Extracting SI-SNR and Squim MOS data
validation_sisnr = [entry[0] for entry in validation_data]
validation_squim_mos = [entry[1] for entry in validation_data]

# Plot configurations
plot_configs = [
    {
        "ax": axs[0],
        "data": validation_sisnr,
        "title": 'Data Fraction Learning Curve (SI-SNR)',
        "ylabel": 'SI-SNR',
    },
    {
        "ax": axs[1],
        "data": validation_squim_mos,
        "title": 'Data Fraction Learning Curve (Squim MOS)',
        "ylabel": 'Squim MOS',
    }
]

# Loop through each subplot configuration
for config in plot_configs:
    ax = config["ax"]
    ax.plot(ticks, config["data"], label=config["title"], marker='o', color='red')
    ax.set_title(config["title"])
    ax.set_xlabel('Percentage of data used for training')
    ax.set_ylabel(config["ylabel"])
    ax.set_xticks(ticks)
    ax.set_xticklabels(['1', '2', '3', '5', '8', '10', '20', '30', '40', '50', '60', '70', '80', '90', '100'])
    for i in range(len(ticks)):
        ax.axvline(x=ticks[i], ymin=0, ymax=(config["data"][i] - ax.get_ylim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0]), color='gray', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig('learning_curves_fraction.png', dpi=300)