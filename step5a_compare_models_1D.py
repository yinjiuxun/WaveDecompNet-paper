from matplotlib import pyplot as plt
import numpy as np
import h5py
from utilities import mkdir
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu
from autoencoder_1D_models_torch import *

from sklearn.metrics import mean_squared_error, explained_variance_score


def signal_to_noise_ratio(signal, noise):
    """Return the SNR in dB"""
    snr0 = np.sum(signal ** 2, axis=0) / (np.sum(noise ** 2, axis=0) + 1e-6)
    snr0 = 10 * np.log10(snr0 + 1e-6)
    return snr0


# make the output directory
output_dir = "./all_model_comparison"
mkdir(output_dir)

# %% load dataset
data_dir = './training_datasets'
data_name = 'training_datasets_STEAD_waveform.hdf5'

# %% load dataset
with h5py.File(data_dir + '/' + data_name, 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# %% Specify the model directory and model name list first
model_dataset_dir = "Model_and_datasets_1D_STEAD"
model_names = ["Autoencoder_Conv1D_None", "Autoencoder_Conv1D_Linear",
               "Autoencoder_Conv1D_LSTM", "Autoencoder_Conv1D_attention"]

model_mse_all = []  # list to store the mse for all models
model_snr_all = []  # list to store the snr for all models

for model_name in model_names:
    model_dir = model_dataset_dir + f'/{model_name}'

    # split the model based on the information provided by the model
    # split the model based on the information provided by the model
    with h5py.File(model_dir + '/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
        train_size = f.attrs['train_size']
        test_size = f.attrs['test_size']
        rand_seed1 = f.attrs['rand_seed1']
        rand_seed2 = f.attrs['rand_seed2']

    _, X_test, _, Y_test = train_test_split(X_train, Y_train,
                                            train_size=train_size, random_state=rand_seed1)
    _, X_test, _, Y_test = train_test_split(X_test, Y_test,
                                            test_size=test_size, random_state=rand_seed2)

    test_data = WaveformDataset(X_test, Y_test)

    # %% load model
    model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())
    print("*" * 12 + " Model " + model_name + " loaded " + "*" * 12)

    batch_size = 256
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Evaluate the test loss for the model
    loss_fn = torch.nn.MSELoss()
    test_loss = 0.0
    model.eval()
    for X, y in test_iter:
        if len(y.data) != batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X)
        # calculate the loss
        loss = loss_fn(output, y)
        # update test loss
        test_loss += loss.item() * X.size(0)

    test_loss = test_loss / len(test_iter.dataset)
    print("*" * 12 + " Model " + model_name + ' Test Loss: {:.6f}\n'.format(test_loss) + "*" * 12)

    # Calculate the MSE distribution for each model
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    model_mse = []
    model_snr = []

    for X, y in test_iter:
        if len(y.data) != batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(X)

        clean_signal = y.detach().numpy()
        clean_signal = np.reshape(clean_signal, (-1, clean_signal.shape[-1]))
        clean_signal = np.moveaxis(clean_signal, -1, 0)

        noise = X.detach().numpy() - y.detach().numpy()
        noise = np.reshape(noise, (-1, noise.shape[-1]))
        noise = np.moveaxis(noise, -1, 0)

        denoised_signal = output.detach().numpy()
        denoised_signal = np.reshape(denoised_signal, (-1, denoised_signal.shape[-1]))
        denoised_signal = np.moveaxis(denoised_signal, -1, 0)

        # calculate the SNR
        snr = signal_to_noise_ratio(clean_signal, noise)
        model_snr.append(snr)

        # calculate the mse between waveforms
        mse = explained_variance_score(clean_signal, denoised_signal, multioutput='raw_values')
        model_mse.append(mse)

    model_mse = np.array(model_mse).flatten()
    model_snr = np.array(model_snr).flatten()

    model_mse_all.append(model_mse)
    model_snr_all.append(model_snr)


import seaborn as sns

plt.close('all')
plt.figure(3)
fig, ax = plt.subplots(2, 2, sharex=True, sharey=True, squeeze=True)
ax = ax.flatten()
i_waveforms = np.random.choice(len(model_mse_all[0]), 10000)

for i, model_name in enumerate(model_names):
    model_mse = model_mse_all[i]
    model_snr = model_snr_all[i]
    ii2 = np.bitwise_and(model_mse <= 1, model_mse >= 0)

    hist, bin_edge = np.histogram(model_mse[ii2], bins=20, range=(0, 1))
    hist = hist / len(model_mse[ii2])
    bin_center = bin_edge[0:-1] + (bin_edge[1] - bin_edge[0]) / 2
    plt.figure(1)
    plt.plot(bin_center, hist, '-o', label=model_name)
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
plt.ylabel('Proportion')
plt.legend()

plt.savefig(output_dir + '/histograms.png')