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
model_dataset_dir = "Model_and_datasets_1D_all_snr_40" + "/all_runs"

model_names = ["Branch_Encoder_Decoder_None", "Branch_Encoder_Decoder_Linear",
               "Branch_Encoder_Decoder_LSTM", "Branch_Encoder_Decoder_attention",
               "Branch_Encoder_Decoder_Transformer"]
bottleneck_names = ["None", "Linear", "LSTM", "Attention", "Transformer"] #

line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22',
               '#17becf', '#ff22b4']

####### plot training curves for all models
# %% Show loss evolution
plt.close('all')
fig, ax = plt.subplots(3, 2, sharey=True, figsize=(11, 13))
ax = ax.flatten()

fig_test, ax_test = plt.subplots(3, 2, sharey=True, figsize=(11, 13))
ax_test = ax_test.flatten()

# Only keep the training history until the best model before early stopping
early_stopping_patience = 10

for i_model, model_name in enumerate(model_names):
    model_test_loss_all = []
    model_test_loss_list_all = []
    for j_model in range(11):
        model_dir = model_dataset_dir + f'/{model_name}' + str(j_model)
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

        plt.figure(fig.number)
        ax[i_model].semilogy(loss, 'o', label=f'Training loss {j_model}',
                             color=line_colors[j_model], alpha=1, markersize=3)
        ax[i_model].semilogy(val_loss, 'x', label=f'Validation loss {j_model}',
                             linewidth=1, color=line_colors[j_model], alpha=0.8, markersize=3)

        if i_model in [0, 2, 4]:
            ax[i_model].set_ylabel('MSE', fontsize=12)
        if i_model in [3, 4]:
            ax[i_model].set_xlabel('Epochs', fontsize=12)
        ax[i_model].grid()
        # ax[i_model].set_title(bottleneck_name, fontsize=14)
        ax[i_model].set_title(bottleneck_name, fontsize=14)


    #     # Check test loss
    #     with h5py.File(model_dir + '/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    #         train_size = f.attrs['train_size']
    #         test_size = f.attrs['test_size']
    #         rand_seed1 = f.attrs['rand_seed1']
    #         rand_seed2 = f.attrs['rand_seed2']
    #
    #     X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
    #                                                               train_size=train_size, random_state=rand_seed1)
    #     X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
    #                                                               test_size=test_size, random_state=rand_seed2)
    #
    #     test_data = WaveformDataset(X_test, Y_test)
    #     # %% load model
    #     model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())
    #     print("*" * 12 + " Model " + model_name + " loaded " + "*" * 12)
    #
    #     batch_size = 256
    #     test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    #
    #     # Evaluate the test loss for the model
    #     loss_fn = torch.nn.MSELoss()
    #     test_loss = 0.0
    #     test_loss_list = []
    #     model.eval()
    #     for X, y in test_iter:
    #         loss_fn_details = torch.nn.MSELoss(reduction='none')
    #         if len(y.data) != batch_size:
    #             break
    #         # forward pass: compute predicted outputs by passing inputs to the model
    #         output1, output2 = model(X)
    #         # calculate the loss
    #         loss = loss_fn(output1, y) + loss_fn(output2, X - y)
    #         loss_all = loss_fn_details(output1, y) + loss_fn_details(output2, X - y)
    #         # update test loss
    #         test_loss += loss.item() * X.size(0)
    #
    #         # store each test values
    #         temp_array = np.mean(np.mean(loss_all.detach().numpy(), axis=-1), axis=-1)
    #         test_loss_list.append(temp_array)
    #
    #     test_loss = test_loss / len(test_iter.dataset)
    #     model_test_loss_all.append(test_loss)
    #     model_test_loss_list_all.append(np.array(test_loss_list).flatten())
    #
    # ax_test[i_model].hist(model_test_loss_list_all, range=(0, 0.5), bins=10)

ax[0].set_ylim(0.04, 1)
ax[-2].legend(fontsize=12, loc=(1.15, 0.0), ncol=2)
ax[-1].set_visible(False)

for i, letter_label in enumerate(['(a)', '(b)', '(c)', '(d)', '(e)']):
    ax[i].annotate(letter_label, xy=(-0.1, 1.05), xycoords=ax[i].transAxes, fontsize=16)

plt.savefig(model_dataset_dir + '/all_run_comparison.pdf', bbox_inches='tight')