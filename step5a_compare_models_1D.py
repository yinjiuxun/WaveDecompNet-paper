from matplotlib import pyplot as plt
import numpy as np
import h5py
from utilities import mkdir
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu, parameter_number
from autoencoder_1D_models_torch import *

from sklearn.metrics import mean_squared_error, explained_variance_score


def signal_to_noise_ratio(signal, noise, axis=0):
    """Return the SNR in dB"""
    snr0 = np.sum(signal ** 2, axis=axis) / (np.sum(noise ** 2, axis=axis) + 1e-6)
    snr0 = 10 * np.log10(snr0 + 1e-6)
    return snr0


# %% load dataset
data_dir = './training_datasets'
data_name = 'training_datasets_all_snr_40.hdf5'
# data_name = 'training_datasets_waveform.hdf5'
# data_name = 'training_datasets_STEAD_plus_POHA.hdf5'

# %% load dataset
with h5py.File(data_dir + '/' + data_name, 'r') as f:
    time = f['time_new'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# %% Specify the model directory and model name list first
model_dataset_dir = "Model_and_datasets_1D_all_snr_40"
# model_dataset_dir = "Model_and_datasets_1D_synthetic"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
# model_names = ["Autoencoder_Conv1D_None", "Autoencoder_Conv1D_Linear",
#                "Autoencoder_Conv1D_LSTM", "Autoencoder_Conv1D_attention",
#                "Autoencoder_Conv1D_attention_LSTM", "Autoencoder_Conv1D_Transformer"]
model_names = ["Branch_Encoder_Decoder_None", "Branch_Encoder_Decoder_Linear",
               "Branch_Encoder_Decoder_LSTM", "Branch_Encoder_Decoder_attention",
               "Branch_Encoder_Decoder_Transformer", "Branch_Encoder_Decoder_hybrid"]
bottleneck_names = ["None", "Linear", "LSTM", "Attention", "Transformer", "Hybrid"]

# make the output directory
output_dir = model_dataset_dir + "/" + "all_model_comparison"
mkdir(output_dir)

model_mse_earthquake_all = []  # list to store the mse for all models
model_mse_noise_all = []  # list to store the noise mse for all models
model_snr_all = []  # list to store the snr for all models
model_test_loss_all = []  # list to store the model test loss
model_test_loss_list_all = []  # List to store all the values of test loss for each model
model_param_number = []  # list to store the number of model parameters

for model_name in model_names:
    model_dir = model_dataset_dir + f'/{model_name}'

    # split the model based on the information provided by the model
    # split the model based on the information provided by the model
    with h5py.File(model_dir + '/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
        train_size = f.attrs['train_size']
        test_size = f.attrs['test_size']
        rand_seed1 = f.attrs['rand_seed1']
        rand_seed2 = f.attrs['rand_seed2']

    X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
                                                              train_size=train_size, random_state=rand_seed1)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                              test_size=test_size, random_state=rand_seed2)

    # first check the SNR distribution in each datasets
    SNR_training = signal_to_noise_ratio(Y_training, X_training - Y_training, axis=1) / 10
    SNR_validate = signal_to_noise_ratio(Y_validate, X_validate - Y_validate, axis=1) / 10
    SNR_test = signal_to_noise_ratio(Y_test, X_test - Y_test, axis=1) / 10
    plt.figure()
    plt.hist([SNR_training.flatten(), SNR_validate.flatten(), SNR_test.flatten()],
             density=True, bins=20, label=['training', 'validate', 'test'])
    plt.xlabel('log10(SNR)', fontsize=16)
    plt.ylabel('Probability density', fontsize=16)
    plt.legend()
    plt.savefig(model_dir + '/SNR_distribution_of_datasets.pdf')
    plt.close('all')

    test_data = WaveformDataset(X_test, Y_test)

    # %% load model
    model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())
    print("*" * 12 + " Model " + model_name + " loaded " + "*" * 12)

    batch_size = 256
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Evaluate the test loss for the model
    loss_fn = torch.nn.MSELoss()
    test_loss = 0.0
    test_loss_list = []
    model.eval()
    for X, y in test_iter:
        loss_fn_details = torch.nn.MSELoss(reduction='none')
        if len(y.data) != batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2 = model(X)
        # calculate the loss
        loss = loss_fn(output1, y) + loss_fn(output2, X - y)
        loss_all = loss_fn_details(output1, y) + loss_fn_details(output2, X - y)
        # update test loss
        test_loss += loss.item() * X.size(0)

        # store each test values
        temp_array = np.mean(np.mean(loss_all.detach().numpy(), axis=-1), axis=-1)
        test_loss_list.append(temp_array)

    test_loss = test_loss / len(test_iter.dataset)
    model_test_loss_all.append(test_loss)
    model_test_loss_list_all.append(np.array(test_loss_list).flatten())
    model_param_number.append(parameter_number(model))
    print("*" * 12 + " Model " + model_name + ' Test Loss: {:.6f}\n'.format(test_loss) + "*" * 12)

    # Calculate the MSE distribution for each model
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model_mse_earthquake = []
    model_mse_noise = []
    model_snr = []

    for X, y in test_iter:
        if len(y.data) != batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2 = model(X)

        clean_signal = y.detach().numpy()
        clean_signal = np.reshape(clean_signal, (-1, clean_signal.shape[-1]))
        clean_signal = np.moveaxis(clean_signal, -1, 0)

        clean_noise = (X - y).detach().numpy()
        clean_noise = np.reshape(clean_noise, (-1, clean_noise.shape[-1]))
        clean_noise = np.moveaxis(clean_noise, -1, 0)

        noise = output2.detach().numpy()
        noise = np.reshape(noise, (-1, noise.shape[-1]))
        noise = np.moveaxis(noise, -1, 0)

        denoised_signal = output1.detach().numpy()
        denoised_signal = np.reshape(denoised_signal, (-1, denoised_signal.shape[-1]))
        denoised_signal = np.moveaxis(denoised_signal, -1, 0)

        # calculate the SNR
        snr = signal_to_noise_ratio(clean_signal, clean_noise)
        model_snr.append(snr)

        # calculate the mse between waveforms
        mse = explained_variance_score(clean_signal, denoised_signal, multioutput='raw_values')
        mse_noise = explained_variance_score(clean_noise, noise, multioutput='raw_values')

        model_mse_earthquake.append(mse)
        model_mse_noise.append(mse_noise)

    model_mse_earthquake = np.array(model_mse_earthquake).flatten()
    model_mse_noise = np.array(model_mse_noise).flatten()
    model_snr = np.array(model_snr).flatten()

    model_mse_earthquake_all.append(model_mse_earthquake)
    model_mse_noise_all.append(model_mse_noise)
    model_snr_all.append(model_snr)

# Save the mse and SNR of all models
# %% Save the pre-processed datasets
model_comparison = output_dir + '/all_model_comparison.hdf5'
with h5py.File(model_comparison, 'w') as f:
    f.create_dataset('model_names', data=model_names)
    f.create_dataset('model_mse_earthquake_all', data=model_mse_earthquake_all)
    f.create_dataset('model_mse_noise_all', data=model_mse_noise_all)
    f.create_dataset('model_snr_all', data=model_snr_all)
    f.create_dataset('model_test_loss_all', data=model_test_loss_all)
    f.create_dataset('model_test_loss_list_all', data=model_test_loss_list_all)
    f.create_dataset('model_param_number', data=model_param_number)

# Load the saved model comparison
from matplotlib import pyplot as plt
import numpy as np
import h5py
import matplotlib
from scipy.stats import mode

matplotlib.rcParams.update({'font.size': 12})
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf']

# %% Specify the model directory and model name list first
model_dataset_dir = "Model_and_datasets_1D_all_snr_40"
# model_dataset_dir = "Model_and_datasets_1D_synthetic"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
output_dir = model_dataset_dir + "/" + "all_model_comparison"

with h5py.File(output_dir + '/all_model_comparison.hdf5', 'r') as f:
    # model_names = f['model_names'][:]
    model_mse_earthquake_all = f['model_mse_earthquake_all'][:]
    model_mse_noise_all = f['model_mse_noise_all'][:]
    model_snr_all = f['model_snr_all'][:]
    model_test_loss_all = f['model_test_loss_all'][:]
    model_test_loss_list_all = f['model_test_loss_list_all'][:]
    model_param_number = f['model_param_number'][:]

model_names = model_names[0:5]  # not show Hybrid model for now.

####### plot training curves for all models
# %% Show loss evolution
plt.close('all')
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(11, 13))
ax = ax.flatten()

# Only keep the training history until the best model before early stopping
early_stopping_patience = 10
for i_model, model_name in enumerate(model_names):
    model_dir = model_dataset_dir + f'/{model_name}'
    test_loss = model_test_loss_all[i_model]
    param_number = model_param_number[i_model]
    bottleneck_name = bottleneck_names[i_model]
    with h5py.File(model_dir + '/' + f'{model_name}_Training_history.hdf5', 'r') as f:
        loss = f['loss'][:][:-early_stopping_patience]
        val_loss = f['val_loss'][:][:-early_stopping_patience]
        earthquake_loss = f['earthquake_loss'][:][:-early_stopping_patience]
        earthquake_val_loss = f['earthquake_val_loss'][:][:-early_stopping_patience]
        noise_loss = f['noise_loss'][:][:-early_stopping_patience]
        noise_val_loss = f['noise_val_loss'][:][:-early_stopping_patience]

        # # there is no validation loss in the first epoch
        # val_loss = np.append(np.nan, val_loss)
        # earthquake_val_loss = np.append(np.nan, earthquake_val_loss)
        # noise_val_loss = np.append(np.nan, noise_val_loss)

    partial_loss = [earthquake_loss, earthquake_val_loss, noise_loss, noise_val_loss]

    ax[i_model].plot(loss, 'ko', label='Training loss')
    ax[i_model].plot(val_loss, 'k-', label='Validation loss', linewidth=2)
    ax[i_model].plot([len(loss)], [test_loss], 'r*', label=f'Test loss', markersize=10, linewidth=2,
                     zorder=10)

    loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
    loss_plot_marker_list = ['o', '', 'o', '']
    loss_plot_line_list = ['', '-', '', '-']
    loss_plot_color_list = ['b', 'b', 'g', 'g']
    for ii in range(4):
        ax[i_model].plot(partial_loss[ii], marker=loss_plot_marker_list[ii],
                         linestyle=loss_plot_line_list[ii],
                         color=loss_plot_color_list[ii],
                         label=loss_name_list[ii])

    if i_model in [0, 2, 4]:
        ax[i_model].set_ylabel('MSE', fontsize=12)
    if i_model in [3, 4]:
        ax[i_model].set_xlabel('Epochs', fontsize=12)
    ax[i_model].grid()
    # ax[i_model].set_title(bottleneck_name, fontsize=14)
    ax[i_model].set_title(bottleneck_name, fontsize=14)

    props = dict(boxstyle='round', facecolor='white', alpha=0.7)
    ax[i_model].text(0.98, 0.84, f'{param_number} parameters \n' + f'Mean test loss {test_loss:.4f}',
                     fontsize=12, transform=ax[i_model].transAxes, horizontalalignment='right',
                     bbox=props)
    plt.show()
ax[-2].legend(fontsize=14, loc=(1.3, 0.2))
ax[-1].set_visible(False)

for i, letter_label in enumerate(['(a)', '(b)', '(c)', '(d)', '(e)']):
    ax[i].annotate(letter_label, xy=(-0.1, 1.05), xycoords=ax[i].transAxes, fontsize=16)
plt.savefig(output_dir + '/bottleneck_comparison_training_curves.pdf', dpi=200, bbox_inches='tight')

####### plot the EV score distribution
# Extract the relation between SNR and EVS
bin_size = 0.5
snr_bin_edge = np.arange(-2, 4, bin_size)


def continous_mode(data, bins=500, range=(-2, 1)):
    hist, bin_edge = np.histogram(data, bins=bins, range=range)
    bin_center = bin_edge[0:-1] + (bin_edge[1] - bin_edge[0]) / 2
    # plt.plot(bin_center, hist)
    return bin_center[np.argmax(hist)]


def extract_snr_vs_evs(model_snr_all, model_mse_all, snr_bin_edge, center_type="median"):
    snr_bin_center = snr_bin_edge + bin_size / 2
    mse_median_all = []
    mse_mean_all = []
    mse_mode_all = []
    mse_std_all = []
    mse_error_bar_all = []

    for i in range(len(model_names)):
        mse_median = []
        mse_mean = []
        mse_mode = []
        mse_std = []
        model_snr = model_snr_all[i] / 10
        model_mse = model_mse_all[i]

        mse_error_positive = []
        mse_error_negative = []

        for bin in snr_bin_edge:
            ii_bin = np.bitwise_and(model_snr <= (bin + bin_size), model_snr >= bin)

            mse_current = model_mse[ii_bin]
            # Median
            mse_median_current = np.median(mse_current)
            mse_median.append(mse_median_current)
            # Mean
            mse_mean_current = np.mean(mse_current)
            mse_mean.append(mse_mean_current)
            # STD (not used here)
            mse_std.append(np.std(mse_current))
            # Statistic mode
            mse_mode_current = continous_mode(mse_current)
            mse_mode.append(mse_mode_current)

            # # Use the mean values above and below the MEAN to inllustrate uncertainty
            # mse_error_positive.append(np.mean(mse_current[mse_current > mse_mean_current]) - mse_mean_current)
            # mse_error_negative.append(mse_mean_current - np.mean(mse_current[mse_current < mse_mean_current]))

            # Use the mean values above and below the MEDIAN to inllustrate uncertainty
            mse_error_positive.append(np.median(mse_current[mse_current > mse_median_current]) - mse_median_current)
            mse_error_negative.append(mse_median_current - np.median(mse_current[mse_current < mse_median_current]))

        mse_median_all.append(np.array(mse_median))
        mse_mean_all.append(np.array(mse_mean))
        mse_mode_all.append(np.array(mse_mode))
        mse_std_all.append(np.array(mse_std))
        mse_error_bar_all.append(np.array([mse_error_negative, mse_error_positive]))
        if center_type == "median":
            mse_center_all = mse_median_all
        elif center_type == "mode":
            mse_center_all = mse_mode_all
    if center_type == "mode":
        mse_error_bar_all = [0, 0, 0, 0, 0]

    return snr_bin_center, mse_center_all, mse_error_bar_all


center_type = 'median'
snr_bin_center, mse_center_all, mse_error_bar_all = extract_snr_vs_evs(model_snr_all,
                                                                       model_mse_earthquake_all, snr_bin_edge,
                                                                       center_type=center_type)
plt.close('all')
fig, ax = plt.subplots(2, 2, figsize=(12, 8), gridspec_kw={'width_ratios': [20, 4]}, sharey='row')
fig.tight_layout()
for i in range(len(model_names)):
    ax[0, 0].errorbar(snr_bin_center + i * 0.04 - 0.1, mse_center_all[i], yerr=mse_error_bar_all[i],
                      marker='s', color=line_colors[i], linewidth=1, linestyle='-',
                      label=bottleneck_names[i], elinewidth=1.5, zorder=3)
    ax[0, 0].set_xlim(-2, 4)
    ax[0, 0].set_ylim(-0.4, 1.1)
ax[0, 0].legend(loc=4)
ax[0, 0].set_xlabel('log10(SNR)', fontsize=14)
ax[0, 0].set_ylabel('EV score', fontsize=14)
ax[0, 0].grid()
ax[0, 0].annotate("(a)", xy=(-0.1, 1), xycoords=ax[0, 0].transAxes, fontsize=20)

ax[0, 1].hist(model_mse_earthquake_all[0:-1,:].T, range=(-1, 1.05), bins=10,
           orientation='horizontal', density=True, color=line_colors[0:5])
ax[0, 1].grid()

# show ambient noise waveforms evs
snr_bin_center, mse_center_all, mse_error_bar_all = extract_snr_vs_evs(model_snr_all,
                                                                       model_mse_noise_all, snr_bin_edge,
                                                                       center_type=center_type)
for i in range(len(model_names)):
    ax[1, 0].errorbar(-(snr_bin_center + i * 0.04 - 0.1), mse_center_all[i], yerr=mse_error_bar_all[i],
                      marker='s', color=line_colors[i], linewidth=1, linestyle='-',
                      label=bottleneck_names[i], elinewidth=1.5, zorder=3)
    ax[1, 0].set_xlim(-4, 2)
    ax[1, 0].set_ylim(-0.4, 1.1)
ax[1, 0].legend(loc=4)
ax[1, 0].set_xlabel('-log10(SNR)', fontsize=14)
ax[1, 0].set_ylabel('EV score', fontsize=14)
ax[1, 0].grid()
ax[1, 0].annotate("(b)", xy=(-0.1, 1), xycoords=ax[1, 0].transAxes, fontsize=20)

ax[1, 1].hist(model_mse_noise_all[0:-1,:].T, range=(-1, 1.05), bins=10,
           orientation='horizontal', density=True, color=line_colors[0:5])
ax[1, 1].grid()
ax[1, 1].set_xlabel('Density', fontsize=14)
plt.savefig(output_dir + '/bottleneck_comparison_' + center_type + '_histograms.pdf', dpi=200, bbox_inches='tight')

######### Plot distribution of test loss #########################
plt.close('all')
plt.figure(figsize=(8, 5))
plt.hist(np.array(model_test_loss_list_all[:-1]).T,
         bins=10, range=(0, 0.5), density=True, label=bottleneck_names[:-1])
plt.legend()
plt.xlabel('MSE loss')
plt.ylabel('Density')
plt.savefig(output_dir + '/bottleneck_comparison_test_loss_histograms.pdf', dpi=200, bbox_inches='tight')








############################## Compare Conv1D and Conv2D models ######################################

# %% Specify the model directory and model name list first
model_dataset_dir = "Model_and_datasets_1D_all"
# model_dataset_dir = "Model_and_datasets_1D_synthetic"
# model_dataset_dir = "Model_and_datasets_1D_STEAD_plus_POHA"
# output_dir = model_dataset_dir + "/" + "all_model_comparison_1D"
output_dir = model_dataset_dir + "/" + "all_model_comparison_branches"

with h5py.File(output_dir + '/all_model_comparison.hdf5', 'r') as f:
    model_names = f['model_names'][:]
    model_mse_all = f['model_mse_all'][:]
    model_snr_all = f['model_snr_all'][:]

output_dir = model_dataset_dir + "/" + "all_model_comparison_2D"
with h5py.File(output_dir + '/all_model_comparison.hdf5', 'r') as f:
    model_names = np.concatenate((model_names, f['model_names'][:]))
    model_mse_all = np.concatenate((model_mse_all, f['model_mse_all'][:]), axis=0)
    model_snr_all = np.concatenate((model_snr_all, f['model_snr_all'][:]), axis=0)

output_dir = model_dataset_dir + "/" + "all_model_comparison_1D_vs_2D"
mkdir(output_dir)
bottleneck_names = ["1D None", "1D Linear", "1D LSTM", "1D attention",
                    "1D LSTM_attention", "1D Transformer",
                    "2D None", "2D Linear", "2D LSTM", "2D attention",
                    "2D Transformer"]

line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf', '#ff99ff']

index_model_to_show = [1, 2, 3, 5, 7, 8, 9, 10]

plt.close('all')
plt.figure(3)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=True)
ax = ax.flatten()
i_waveforms = np.random.choice(len(model_mse_all[0]), 10000)

for i, model_name in enumerate(range(len(index_model_to_show))):
    model_mse = model_mse_all[index_model_to_show[i]]
    model_snr = model_snr_all[index_model_to_show[i]]
    bottleneck_name = bottleneck_names[index_model_to_show[i]]
    ii2 = np.bitwise_and(model_mse <= 1, model_mse >= -1)

    hist, bin_edge = np.histogram(model_mse[ii2], bins=20, range=(-1, 1))
    hist = hist / len(model_mse[ii2])
    bin_center = bin_edge[0:-1] + (bin_edge[1] - bin_edge[0]) / 2
    plt.figure(1, figsize=(8, 4))
    plt.plot(bin_center, hist, '-o', color=line_colors[index_model_to_show[i]], label=bottleneck_name)
    #
    # plt.figure(2)
    # ii1 = np.bitwise_and(model_snr <= 20, model_snr >= -40)
    # ii2 = np.bitwise_and(model_mse <= 1, model_mse >= 0)
    # sns.violinplot(x=model_name, y=model_mse[ii2], alpha=0.5)
    #
    #
    # plt.figure(3)
    # ax[i].plot(model_snr, model_mse, '.', label=model_name, alpha=0.1)
    # ax[i].set_ylim(0, 1)
    # ax[i].set_xlim(-40, 20)
    #
    #

plt.figure(1)
plt.xlabel('Explained variance score')
plt.ylabel('Probability Density')
plt.legend()
plt.grid()

plt.savefig(output_dir + '/histograms.pdf')
plt.savefig(output_dir + '/histograms.png', dpi=200, bbox_inches='tight')

# Extract the relation between SNR and EVS
bin_size = 0.5
snr_bin_edge = np.arange(-2, 1, bin_size)
snr_bin_center = snr_bin_edge + bin_size / 2
mse_median_all = []
mse_mean_all = []
mse_std_all = []
for i in range(len(index_model_to_show)):
    mse_median = []
    mse_mean = []
    mse_std = []
    model_snr = model_snr_all[index_model_to_show[i]] / 10
    model_mse = model_mse_all[index_model_to_show[i]]
    for bin in snr_bin_edge:
        ii_bin = np.bitwise_and(model_snr <= (bin + bin_size), model_snr >= bin)
        mse_median.append(np.median(model_mse[ii_bin]))
        mse_mean.append(np.mean(model_mse[ii_bin]))
        mse_std.append(np.std(model_mse[ii_bin]))

    mse_median_all.append(np.array(mse_median))
    mse_mean_all.append(np.array(mse_median))
    mse_std_all.append(np.array(mse_std))

plt.figure(2, figsize=(8, 4))
for i in range(len(index_model_to_show)):
    ii = index_model_to_show[i]
    plt.errorbar(snr_bin_center + i * 0.025 - 0.1, mse_mean_all[i], yerr=mse_std_all[i],
                 marker='s', color=line_colors[i], linewidth=2,
                 label=bottleneck_names[ii], elinewidth=1.5, zorder=3)
    plt.xlim(-2, 1)
    plt.ylim(-0.6, 1.12)
plt.legend(loc=4)
plt.xlabel('log10(SNR)', fontsize=15)
plt.ylabel('Explained Variance', fontsize=15)
plt.grid()

# plt.savefig(output_dir + '/SNR_vs_EV.pdf')
plt.savefig(output_dir + '/SNR_vs_EV.png', dpi=200, bbox_inches='tight')
