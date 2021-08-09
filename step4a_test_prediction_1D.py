from matplotlib import pyplot as plt
import numpy as np
import h5py
from utilities import mkdir
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu

# %% Need to specify model_name first
model_name = 'Autoencoder_conv1d_pytorch'
model_dir = './Model_and_datasets_1D_STEAD' + f'/{model_name}'
data_dir = './training_datasets'

# %% load dataset
with h5py.File(data_dir + '/training_datasets_STEAD_waveform.hdf5', 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# split the model based on the information provided by the model
# split the model based on the information provided by the model
with h5py.File(model_dir + '/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    train_size = f.attrs['train_size']
    test_size = f.attrs['test_size']
    rand_seed1 = f.attrs['rand_seed1']
    rand_seed2 = f.attrs['rand_seed2']

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=train_size, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=test_size, random_state=rand_seed2)

training_data = WaveformDataset(X_train, Y_train)
validate_data = WaveformDataset(X_validate, Y_validate)
test_data = WaveformDataset(X_test, Y_test)

# %% load model
model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())

batch_size = 256
test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=True)

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

test_loss = test_loss/len(test_iter.dataset)
print('Test Loss: {:.6f}\n'.format(test_loss))


# Output some figures
figure_dir = model_dir + '/Figures'
mkdir(figure_dir)

# %% Show loss evolution
with h5py.File(model_dir + '/' + f'{model_name}_Training_history.hdf5', 'r') as f:
    loss = f['loss'][:]
    val_loss = f['val_loss'][:]
plt.figure()
plt.plot(loss, 'o', label='Training loss')
plt.plot(val_loss, '-', label='Validation loss')
plt.plot([1, len(loss)], [test_loss, test_loss], '-', label='Test loss', linewidth=4)
plt.legend()
plt.title(model_name)
#plt.show()
plt.savefig(figure_dir + f"/{model_name}_Loss_evolution.png")


# %% predict the waveforms
# obtain one batch of test images
data_iter = iter(test_iter)
noisy_signal, clean_signal = data_iter.next()

# get sample outputs
denoised_signal = model(noisy_signal)
denoised_signal = denoised_signal.detach().numpy()

# %% Check the waveforms
plt.close("all")
i_model = np.random.randint(0, denoised_signal.shape[0])
fig, ax = plt.subplots(denoised_signal.shape[1], 2, sharex=True, sharey=True, num=1, figsize=(12, 8))

vmax = None
vmin = None
for i in range(noisy_signal.shape[1]):
    ax[i, 0].plot(time, noisy_signal[i_model, i, :], '-k', label='X_input', linewidth=1.5)
    ax[i, 0].plot(time, clean_signal[i_model, i, :], '-r', label='Y_true', linewidth=1)
    ax[i, 1].plot(time, clean_signal[i_model, i, :], '-r', label='Y_true', linewidth=1)
    ax[i, 1].plot(time, denoised_signal[i_model, i, :], '-b', label='Y_predict', linewidth=1)


titles = ['E', 'N', 'Z']
for i in range(noisy_signal.shape[1]):
    ax[i, 0].set_ylabel(titles[i])

ax[0, 0].legend()
ax[0, 1].legend()
ax[-1, 0].set_xlabel('Time (s)')
ax[-1, 1].set_xlabel('Time (s)')
#plt.show()

plt.figure(1)
plt.savefig(figure_dir + f'/{model_name}_Prediction_waveform_model_{i_model}.png')

# # TODO: quantify the model performance from waveform correlation
# norm_Y_test = np.linalg.norm(Y_test, axis=1)
# norm_Y_predict = np.linalg.norm(Y_predict, axis=1)
# corr_coef = np.sum(Y_test * Y_predict, axis=1) / (norm_Y_test + 1e-16) / (norm_Y_predict + 1e-16)
#
# # get the SNR
# noise_std = np.std((X_test - Y_test), axis=1)
# signal_std = np.std(Y_test, axis=1)
# denoised_signal_std = np.std(Y_predict, axis=1)
#
# SNR_before = 10 * np.log10(signal_std / (noise_std + 1e-16))
# SNR_after = 10 * np.log10(denoised_signal_std / (noise_std + 1e-16))
#
# # Maximum amplitude change
# max_amplitude_signal = np.max(Y_test, axis=1)
# max_amplitude_denoise = np.max(Y_predict, axis=1)
# amplitude_change = np.abs(max_amplitude_denoise - max_amplitude_signal) / (max_amplitude_denoise + 1e-16 )
#
# plt.close('all')
# # plt.figure()
# # plt.plot(SNR_before.flatten(), corr_coef.flatten(), '.')
# # plt.ylabel('Corr. Coef.')
# # plt.xlabel('SNR before denoising')
# # plt.xlim(-50, 50)
#
# fig, axi = plt.subplots(3, 1, figsize=(6, 10), sharex=True)
# axi[0].plot(SNR_before.flatten(), SNR_after.flatten(), '.')
# axi[0].set_ylabel('SNR after denoising')
# axi[0].set_ylim(-20, 20)
#
# axi[1].plot(SNR_before.flatten(), corr_coef.flatten(), '.')
# axi[1].set_ylabel('Corr. Coef.')
#
# axi[2].plot(SNR_before.flatten(), amplitude_change.flatten(), '.')
# axi[2].set_ylabel('Max amplitude change')
# axi[2].set_xlabel('SNR before denoising')
#
# plt.xlim(-20, 20)
#


