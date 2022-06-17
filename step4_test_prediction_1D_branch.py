#%%
from matplotlib import pyplot as plt
import numpy as np
import h5py
from utilities import mkdir, waveform_fft
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader
from torch_tools import WaveformDataset, try_gpu, parameter_number
from autoencoder_1D_models_torch import *

import matplotlib
matplotlib.rcParams.update({'font.size': 10})

# %% load dataset
data_dir = '/kuafu/yinjx/WaveDecompNet_dataset/training_datasets'
#data_name = 'training_datasets_STEAD_waveform.hdf5'
#data_name = 'training_datasets_STEAD_plus_POHA.hdf5'
data_name = 'training_datasets_all_snr_40_unshuffled.hdf5'
#data_name = 'training_datasets_waveform.hdf5'

# %% load dataset
with h5py.File(data_dir + '/' + data_name, 'r') as f:
    time = f['time_new'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

#bottleneck_name_list = ['None', 'Linear', 'LSTM', 'attention', 'Transformer']
bottleneck_name_list = ['None', 'Linear', 'attention', 'Transformer']
#%%
for bottleneck_name in bottleneck_name_list:
    plt.close('all')
    # %% Need to specify model_name first
    # model_dataset_dir = "Model_and_datasets_1D_all_snr_40_unshuffled"
    model_dataset_dir = "Model_and_datasets_1D_all_snr_40_unshuffled_equal_epoch100"
    #model_name = "Autoencoder_Conv2D_" + bottleneck_name
    model_name = "Branch_Encoder_Decoder_" + bottleneck_name

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

    training_data = WaveformDataset(X_training, Y_training)
    validate_data = WaveformDataset(X_validate, Y_validate)
    test_data = WaveformDataset(X_test, Y_test)

    # %% load model
    model = torch.load(model_dir + '/' + f'{model_name}_Model.pth', map_location=try_gpu())
    model = model.to('cpu')

    batch_size = 256
    test_iter = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Evaluate the test loss for the model
    loss_fn = torch.nn.MSELoss()
    test_loss = 0.0
    model.eval()
    for X, y in test_iter:
        # X, y = X.to(try_gpu()), y.to(try_gpu())
        if len(y.data) != batch_size:
            break
        # forward pass: compute predicted outputs by passing inputs to the model
        output1, output2 = model(X)
        # calculate the loss
        loss = loss_fn(output1, y) + loss_fn(output2, X - y)
        # update test loss
        test_loss += loss.item() * X.size(0)

    test_loss = test_loss/len(test_iter.dataset)
    print('Test Loss: {:.6f}\n'.format(test_loss))

    param_number = parameter_number(model)
    print(f'Total number of parameters: {param_number}\n')


    # Output some figures
    figure_dir = model_dir + '/Figures'
    mkdir(figure_dir)

    # %% Show loss evolution
    with h5py.File(model_dir + '/' + f'{model_name}_Training_history.hdf5', 'r') as f:
        loss = f['loss'][:]
        val_loss = f['val_loss'][:]
        earthquake_loss = f['earthquake_loss'][:]
        earthquake_val_loss = f['earthquake_val_loss'][:]
        noise_loss = f['noise_loss'][:]
        noise_val_loss = f['noise_val_loss'][:]

    partial_loss = [earthquake_loss, earthquake_val_loss, noise_loss, noise_val_loss]

    plt.close('all')
    plt.figure()
    plt.plot(loss, 'ko', label='Training loss')
    plt.plot(val_loss, 'k-', label='Validation loss', linewidth=2)
    plt.plot([len(loss)], [test_loss], 'r*', label=f'Test loss = {test_loss:.4f}', markersize=10, linewidth=2, zorder=10)

    loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
    loss_plot_marker_list = ['o', '', 'o', '']
    loss_plot_line_list = ['', '-', '', '-']
    loss_plot_color_list = ['b', 'b', 'g', 'g']
    for ii in range(4):
        plt.plot(partial_loss[ii], marker=loss_plot_marker_list[ii],
                 linestyle=loss_plot_line_list[ii],
                 color=loss_plot_color_list[ii],
                 label=loss_name_list[ii])

    plt.legend(fontsize=12)
    plt.ylabel('MSE', fontsize=14)
    plt.xlabel('Epochs', fontsize=14)
    plt.grid()
    plt.title(bottleneck_name + f' ({param_number} params.),' + f' test loss {test_loss:.4f}', fontsize=14)
    #plt.show()

    plt.savefig(figure_dir + f"/{model_name}_Loss_evolution.pdf", bbox_inches='tight')


    # %% predict the waveforms
    # obtain one batch of test images
    data_iter = iter(test_iter)
    noisy_signal, clean_signal = data_iter.next()
    # noisy_signal, clean_signal = noisy_signal.to(try_gpu()), clean_signal.to(try_gpu())

    # get sample outputs
    denoised_signal, separated_noise = model(noisy_signal)

    # Convert tensor to numpy
    noisy_signal = noisy_signal.detach().numpy()
    clean_signal = clean_signal.detach().numpy()
    denoised_signal = denoised_signal.detach().numpy()
    separated_noise = separated_noise.detach().numpy()
    true_noise = noisy_signal - clean_signal

    from sklearn.metrics import mean_squared_error, explained_variance_score

    # %% Check the waveforms
    i_model = np.random.randint(0, denoised_signal.shape[0])
    for i_model in range(100):
        print(i_model)
        plt.close("all")

        fig, ax = plt.subplots(denoised_signal.shape[1], 3, sharex=True, sharey=True, num=1, figsize=(9, 3)) #16, 8

        vmax = None
        vmin = None
        for i in range(noisy_signal.shape[1]):
            scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))

            ax[i, 0].plot(time, noisy_signal[i_model, i, :]/scaling_factor, '-k', label='Noisy signal', linewidth=1.5)
            ax[i, 0].plot(time, clean_signal[i_model, i, :]/scaling_factor, '-r', label='True signal', linewidth=1)
            ax[i, 1].plot(time, clean_signal[i_model, i, :]/scaling_factor, '-r', label='True signal', linewidth=1)
            ax[i, 1].plot(time, denoised_signal[i_model, i, :]/scaling_factor, '-b', label='Predicted signal', linewidth=1)
            # explained variance score
            evs_earthquake = explained_variance_score(clean_signal[i_model, i, :], denoised_signal[i_model, i, :])
            ax[i, 1].text(50, 0.8, f'EV: {evs_earthquake:.2f}')


            noise = separated_noise[i_model, i, :]
            evs_noise = explained_variance_score(true_noise[i_model, i, :], noise)
            ax[i, 2].plot(time, true_noise[i_model, i, :] / scaling_factor, '-', color='gray', label='True noise', linewidth=1)
            ax[i, 2].plot(time, noise / scaling_factor, '-b',  label='Predicted noise', linewidth=1)
            ax[i, 2].text(50, 0.8, f'EV: {evs_noise:.2f}')

        ax[0, 0].set_title("Original signal")
        if bottleneck_name == "Transformer":
            ax[0, 1].set_title("Earthquake signal (Trans.)")
            ax[0, 2].set_title("Separated noise (Trans.)")
        else:
            ax[0, 1].set_title("Earthquake signal (" + bottleneck_name + ")")
            ax[0, 2].set_title("Separated noise (" + bottleneck_name + ")")

        titles = ['E', 'N', 'Z']
        for i in range(noisy_signal.shape[1]):
            ax[i, 0].set_ylabel(titles[i])

        for i in range(3):
            for j in range(3):
                #ax[i, j].axis('off')
                ax[i, j].xaxis.set_visible(False)
                ax[i, j].yaxis.set_ticks([])
                # remove box
                ax[i, j].spines['right'].set_visible(False)
                ax[i, j].spines['left'].set_visible(False)
                ax[i, j].spines['top'].set_visible(False)
                ax[i, j].spines['bottom'].set_visible(False)

                if i == 2:
                    ax[i, j].xaxis.set_visible(True)
                    ax[i, j].set_xlim(0, 60)
                    ax[i, j].spines['bottom'].set_visible(True)



        #ax[0, 0].legend()
        #ax[0, 1].legend()
        ax[-1, 0].set_xlabel('Time (s)')
        ax[-1, 1].set_xlabel('Time (s)')
        ax[-1, 2].set_xlabel('Time (s)')
        # plt.show()

        plt.figure(1)
        plt.savefig(figure_dir + f'/{model_name}_Prediction_waveform_model_{i_model}.pdf',
                    bbox_inches='tight')

    # %% Check the waveform spectrum
    for i_model in range(100):
        print(i_model)
        plt.close("all")

        fig, ax = plt.subplots(2, denoised_signal.shape[1], sharex=True, sharey=True, num=1, figsize=(12, 5)) #16, 8

        for i in range(noisy_signal.shape[1]):
            noise = separated_noise[i_model, i, :]  # separated noise from ML
            original_noise = noisy_signal[i_model, i, :] - clean_signal[i_model, i, :] # original noise signal
            scaling_factor = np.max(abs(noisy_signal[i_model, i, :]))
            dt = time[1] - time[0]
            _, spect_noisy_signal = waveform_fft(noisy_signal[i_model, i, :]/scaling_factor, dt)
            _, spect_clean_signal = waveform_fft(clean_signal[i_model, i, :]/scaling_factor, dt)
            _, spect_noise = waveform_fft(noise / scaling_factor, dt)
            _, spect_original_noise = waveform_fft(original_noise / scaling_factor, dt)
            freq, spect_denoised_signal = waveform_fft(denoised_signal[i_model, i, :]/scaling_factor, dt)

            #ax[i].loglog(freq, spect_noisy_signal, '-k', label='X_input', linewidth=1.5)
            ax[1, i].loglog(freq, spect_noise, '-', color='b', label='noise', linewidth=0.5, alpha=0.8)
            ax[1, i].loglog(freq, spect_original_noise, 'r', label='orginal noise', linewidth=0.5, alpha=1)

            ax[0, i].loglog(freq, spect_noisy_signal, '-k', label='raw signal', linewidth=0.5, alpha=1)
            ax[0, i].loglog(freq, spect_clean_signal, '-r', label='true earthquake', linewidth=0.5, alpha=1)
            ax[0, i].loglog(freq, spect_denoised_signal, '-b', label='separated earthquake', linewidth=0.5, alpha=1)

            ax[0, i].grid(alpha=0.2)
            if i == 0:
                #ax[0, i].set_xlabel('Frequency (Hz)')
                ax[0, i].set_ylabel('velocity spectra', fontsize=14)
                ax[1, i].set_ylabel('velocity spectra', fontsize=14)

            ax[1, i].grid(alpha=0.2)
            ax[1, i].set_xlabel('Frequency (Hz)', fontsize=14)


        #ax[0, i].legend()
        #ax[1, i].legend()
        # ax[0, 0].set_title("Original signal")
        # if bottleneck_name == "Transformer":
        #     ax[0, 1].set_title("Earthquake signal (Trans.)")
        #     ax[0, 2].set_title("Separated noise (Trans.)")
        # else:
        #     ax[0, 1].set_title("Earthquake signal (" + bottleneck_name + ")")
        #     ax[0, 2].set_title("Separated noise (" + bottleneck_name + ")")

        titles = ['E', 'N', 'Z']
        for i in range(noisy_signal.shape[1]):
            ax[0, i].set_title(titles[i], fontsize=16)
            #ax[1, i].set_title(titles[i])

        plt.figure(1)
        plt.savefig(figure_dir + f'/{model_name}_spectrum_{i_model}.pdf',
                    bbox_inches='tight')

# %%
