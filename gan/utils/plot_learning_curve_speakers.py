import matplotlib.pyplot as plt
import altair as alt
import pandas as pd

fractions_of_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 28]

# Initialize lists for storing SI-SNR and Squim MOS values
# Lists are formatted as: [Ordering 1, Ordering 2, Ordering 3]
val_sisnr = [
    [13.9153144159942, 13.189293809886100, 11.10213214036040],  # 1 Speaker
    [14.906119079555100, 15.41700223724820, 15.165030516467000],  # 2 Speakers
    [16.197081410190400, 16.605519061528400, 16.27967796105780],  # 3 Speakers
    [16.327924684847400, 17.1632645066502, 16.4665124286147],  # 4 Speakers
    [16.270611980875700, 17.198090776077800, 16.63455162117780],  # 5 Speakers
    [16.190076402090100, 17.313562362807500, 16.6997183298023],  # 6 Speakers
    [16.40012998864490, 17.26300370519600, 16.843865058375800],  # 7 Speakers
    [16.187706014485000, 16.980047137702600, 16.88883875816770],  # 8 Speakers
    [15.513570365396500, 16.68188836302570, 16.947607034907800],  # 9 Speakers
    [15.493729346585500, 16.742680402924700, 16.73540992904640],  # 10 Speakers
    [16.495837020063900, 16.87684010678130, 16.966102259830400],  # 15 Speakers
    [16.289068239406500, 16.914295746863500, 16.82581686192350],  # 20 Speakers
    [16.627839125186500, 16.896732028826900, 16.828513745254700],  # 25 Speakers
    [16.828539498078000, 16.742823031342100, 16.742823031342100]   # 28 Speakers
]

val_squim_mos = [
    [3.860970986699600, 3.7947750919073500, 3.7890711284956900],  # 1 Speaker
    [3.6844534431267700, 3.8820226855069700, 3.781502538514370],  # 2 Speakers
    [3.927155478197390, 3.91961699984606, 3.9020265158519000],  # 3 Speakers
    [3.9351816504325700, 3.9297181200055200, 3.9104189305629500],  # 4 Speakers
    [3.950356648384950, 3.9962740867462000, 3.945188513658580],  # 5 Speakers
    [4.038887495554770, 3.992377680482220, 3.910675531162800],  # 6 Speakers
    [3.938115499262670, 4.026432449956540, 3.9226014116435400],  # 7 Speakers
    [3.9703056560558000, 3.9592953442948500, 3.9387088206786600],  # 8 Speakers
    [3.853456943937880, 4.02619443935098, 3.972253864540640],  # 9 Speakers
    [3.9355906285707200, 4.003380749989480, 3.978816347793470],  # 10 Speakers
    [3.921639100151160, 4.049604207277300, 4.024022574852970],  # 15 Speakers
    [3.9694400669880300, 4.034409024472380, 3.9897704416687000],  # 20 Speakers
    [3.9411822985676900, 4.048992604017260, 4.011498986517340],  # 25 Speakers
    [3.9984980891051800, 3.9335797440658500, 3.9390399033583500]   # 28 Speakers
]


# Calculate means
validation_sisnr = [sum(vals) / len(vals) for vals in val_sisnr]
validation_squim_mos = [sum(vals) / len(vals) for vals in val_squim_mos]

# # Creating subplots
# fig, axs = plt.subplots(1, 2, figsize=(20, 6))  # 1 row, 2 columns

# # SI-SNR subplot
# axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
# axs[0].set_title('Speaker-separated Learning Curve (SI-SNR)')
# axs[0].set_xlabel('Number of speakers used for training')
# axs[0].set_ylabel('SI-SNR')
# axs[0].set_xticks(fractions_of_data)

# # Squim MOS subplot
# axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
# axs[1].set_title('Speaker-separated Learning Curve (Squim MOS)')
# axs[1].set_xlabel('Number of speakers used for training')
# axs[1].set_ylabel('Squim MOS')
# axs[1].set_xticks(fractions_of_data)

# plt.savefig('learning_curves.png', dpi=300)
# plt.show()

def create_plots(fractions_of_data, validation_sisnr, validation_squim_mos):
    fig, axs = plt.subplots(1, 2, figsize=(20*0.7, 6*0.7))

    # SI-SNR subplot
    axs[0].plot(fractions_of_data, validation_sisnr, label='Validation SI-SNR', marker='o', color='red')
    axs[0].set_title('Speaker-separated Learning Curve (SI-SNR)')
    axs[0].set_xlabel('Number of speakers used for training')
    axs[0].set_ylabel('SI-SNR')
    axs[0].set_xticks(fractions_of_data)
    axs[0].grid(axis='x')
    
    # Squim MOS subplot
    axs[1].plot(fractions_of_data, validation_squim_mos, label='Validation Squim MOS', marker='o', color='red')
    axs[1].set_title('Speaker-separated Learning Curve (Squim MOS)')
    axs[1].set_xlabel('Number of speakers used for training')
    axs[1].set_ylabel('Squim MOS')
    axs[1].set_xticks(fractions_of_data)
    axs[1].grid(axis='x')
    
    # # Add vertical lines (SI-SNR) - Corrected
    # for x, y in zip(fractions_of_data, validation_sisnr):
    #     axs[0].vlines(x, 12.5, y, colors='white', linestyles='dashed')

    # # Add vertical lines (Squim MOS) - Corrected
    # for x, y in zip(fractions_of_data, validation_squim_mos):
    #     axs[1].vlines(x, 3.75, y, colors='gray', linestyles='dashed')

    # Create Altair chart for SI-SNR
    chart1 = alt.Chart(pd.DataFrame({'x': fractions_of_data, 'y': validation_sisnr})).mark_line(point=True).encode(
        x='x',
        y='y',
        tooltip=['x', 'y']
    ).properties(
        title='Speaker-separated Learning Curve (SI-SNR)'
    ).interactive()

    chart1.save('si_snr_learning_curve.json')

    # Create Altair chart for Squim MOS
    chart2 = alt.Chart(pd.DataFrame({'x': fractions_of_data, 'y': validation_squim_mos})).mark_line(point=True).encode(
        x='x',
        y='y',
        tooltip=['x', 'y']
    ).properties(
        title='Speaker-separated Learning Curve (Squim MOS)'
    ).interactive()

    chart2.save('squim_mos_learning_curve.json')

    plt.savefig('learning_curves.png', dpi=300)
    # plt.show()

create_plots(fractions_of_data, validation_sisnr, validation_squim_mos)