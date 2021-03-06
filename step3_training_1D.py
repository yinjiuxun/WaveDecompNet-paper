#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 10:55:02 2021

@author: Yin9xun
"""
# %%
import os
from tabnanny import verbose
from matplotlib import pyplot as plt
import numpy as np
import h5py
from sklearn.model_selection import train_test_split
import copy

# utility module
from utilities import mkdir

import torch
from torch_tools import WaveformDataset, EarlyStopping, try_gpu, training_loop, training_loop_branches
from torch.utils.data import DataLoader
from autoencoder_1D_models_torch import Autoencoder_Conv1D, Autoencoder_Conv2D, Attention_bottleneck, \
    Attention_bottleneck_LSTM, SeismogramEncoder, SeismogramDecoder, SeisSeparator

def write_running_progress(file_name, text_contents):
    # Recording the running progress to a text file
    with open(file_name, 'a') as f:
        f.write(text_contents)


# make the output directory
model_dataset_dir = './Model_and_datasets_1D_all_snr_40_unshuffled_equal_epoch100'
mkdir(model_dataset_dir)
# progress text file name
progress_file_name = model_dataset_dir + '/Running_progress.txt'

# Choose different model structure: for now Conv1D and Conv2D
model_structure = "Branch_Encoder_Decoder"  # "Autoencoder_Conv1D", "Autoencoder_Conv2D", "Branch_Encoder_Decoder_LSTM"

# Choose a bottleneck type
bottleneck_names = ["None", "Linear", "LSTM", "attention", "Transformer"]

for bottleneck_name in bottleneck_names:

    # Recording the running progress to a text file
    write_running_progress(progress_file_name,
                           text_contents = "=" * 5 + " Working on " + bottleneck_name + "=" * 5  + '\n')

    # %% Read the pre-processed datasets
    print("#" * 12 + " Loading data " + "#" * 12)
    model_datasets = '/kuafu/yinjx/WaveDecompNet_dataset/training_datasets/training_datasets_all_snr_40_unshuffled.hdf5'
    with h5py.File(model_datasets, 'r') as f:
        X_train = f['X_train'][:]
        Y_train = f['Y_train'][:]

    # 3. split to training (60%), validation (20%) and test (20%)
    train_size = 0.6
    test_size = 0.5
    rand_seed1 = 13
    rand_seed2 = 20
    X_training, X_test, Y_training, Y_test = train_test_split(X_train, Y_train,
                                                              train_size=train_size, random_state=rand_seed1)
    X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test,
                                                              test_size=test_size, random_state=rand_seed2)
    for i_run in ['_warmup']:  # run the model multiple times to ensure results are stable.

        # Recording the running progress to a text file
        write_running_progress(progress_file_name,
                               text_contents="+" * 5 + " Working on " + str(i_run) + "th run " + "+" * 5 + '\n')

        # Give a fixed seed for model initialization
        torch.manual_seed(99)
        import random
        random.seed(0)
        np.random.seed(20)
        torch.backends.cudnn.benchmark = False

        # Convert to the dataset class for Pytorch (here simply load all the data,
        # but for the sake of memory, can also use WaveformDataset_h5)
        training_data = WaveformDataset(X_training, Y_training)
        validate_data = WaveformDataset(X_validate, Y_validate)

        if bottleneck_name == "None":
            # Model without specified bottleneck
            bottleneck = None

        elif bottleneck_name == "Linear":
            # Linear bottleneck
            bottleneck = torch.nn.Linear(64, 64, dtype=torch.float64)

        elif bottleneck_name == "LSTM":
            # The encoder-decoder model with LSTM bottleneck
            bottleneck = torch.nn.LSTM(64, 32, 2, bidirectional=True,
                                       batch_first=True, dtype=torch.float64)

        elif bottleneck_name == "attention":
            # Self-attention bottleneck
            #bottleneck = Attention_bottleneck(64, 4, 0.2)  # Add the attention bottleneck
            bottleneck = torch.nn.MultiheadAttention(64, 4, 0.2, 
                                                    batch_first=True, dtype=torch.float64)

        elif bottleneck_name == "Transformer":
            # The encoder-decoder model with transformer encoder as bottleneck
            encoder_layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=4, dtype=torch.float64)
            bottleneck = torch.nn.TransformerEncoder(encoder_layer, num_layers=1)

        elif bottleneck_name == "attention_LSTM":
            # Attention bottleneck with LSTM as positional embedding
            bottleneck = Attention_bottleneck_LSTM(64, 4, 0.2)  # Add the attention bottleneck

        elif bottleneck_name == "hybrid":
            # LSTM for earthquake and Attention bottleneck for noise
            bottleneck1 = torch.nn.LSTM(64, 32, 2, bidirectional=True,
                                       batch_first=True, dtype=torch.float64)
            bottleneck2 = Attention_bottleneck(64, 4, 0.2)  # Add the attention bottleneck

        else:
            raise NameError('bottleneck type has not been defined!')

        # Give a name to the network
        model_name = model_structure + "_" + bottleneck_name
        print("#" * 12 + " building model " + model_name + " " + "#" * 12)


        # Set up model network
        if model_structure == "Autoencoder_Conv1D":
            model = Autoencoder_Conv1D(model_name, bottleneck).to(device=try_gpu())

        elif model_structure == "Autoencoder_Conv2D":
            model = Autoencoder_Conv2D(model_name, bottleneck).to(device=try_gpu())

        elif model_structure == "Branch_Encoder_Decoder":
            if bottleneck_name == "hybrid":
                bottleneck_earthquake = bottleneck1
                bottleneck_noise = bottleneck2
            else:
                bottleneck_earthquake = copy.deepcopy(bottleneck)
                bottleneck_noise = copy.deepcopy(bottleneck)

            encoder = SeismogramEncoder()
            decoder_earthquake = SeismogramDecoder(bottleneck=bottleneck_earthquake)
            decoder_noise = SeismogramDecoder(bottleneck=bottleneck_noise)

            model = SeisSeparator(model_name, encoder, decoder_earthquake, decoder_noise).to(device=try_gpu())

        else:
            raise NameError('Network structure has not been defined!')

        # make the output directory to store the model information
        model_dataset_dir_current = model_dataset_dir + '/' + model_name + str(i_run)
        mkdir(model_dataset_dir_current)

        batch_size, epochs, lr = 128, 100, 1e-3
        minimum_epochs = 30  # the minimum epochs that the training has to do
        patience = None  # patience of the early stopping

        loss_fn = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        #LR_func = lambda epoch: 0.95**epoch if (epoch <50) else 0.95**50
        LR_func = lambda epoch: (epoch+1)/11 if (epoch<=10) else (0.95**epoch if (epoch <50) else 0.95**50) # with warmup
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=LR_func, verbose=True)
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95, verbose=True)
        train_iter = DataLoader(training_data, batch_size=batch_size, shuffle=True) # need to be shuffled
        validate_iter = DataLoader(validate_data, batch_size=batch_size, shuffle=False) # don't shuffle

        print("#" * 12 + " training model " + model_name + " " + "#" * 12)

        if model_structure == "Branch_Encoder_Decoder":
            model, avg_train_losses, avg_valid_losses, partial_loss = training_loop_branches(train_iter, validate_iter,
                                                                                             model, loss_fn, optimizer, scheduler,
                                                                                             epochs=epochs, patience=patience,
                                                                                             device=try_gpu(),
                                                                                             minimum_epochs=minimum_epochs)
        else:
            model, avg_train_losses, avg_valid_losses = training_loop(train_iter, validate_iter,
                                                                      model, loss_fn, optimizer,
                                                                      epochs=epochs, patience=patience, device=try_gpu())
        print("Training is done!")

        # Recording the running progress to a text file
        write_running_progress(progress_file_name,
                               text_contents="Training is done!" + '\n')

        # %% Save the model
        torch.save(model, model_dataset_dir_current + f'/{model_name}_Model.pth')

        loss = avg_train_losses
        val_loss = avg_valid_losses
        # store the model training history
        with h5py.File(model_dataset_dir_current + f'/{model_name}_Training_history.hdf5', 'w') as f:
            f.create_dataset("loss", data=loss)
            f.create_dataset("val_loss", data=val_loss)
            if model_structure == "Branch_Encoder_Decoder":
                f.create_dataset("earthquake_loss", data=partial_loss[0])
                f.create_dataset("earthquake_val_loss", data=partial_loss[1])
                f.create_dataset("noise_loss", data=partial_loss[2])
                f.create_dataset("noise_val_loss", data=partial_loss[3])

        # add some model information
        with h5py.File(model_dataset_dir_current + f'/{model_name}_Dataset_split.hdf5', 'w') as f:
            f.attrs['model_name'] = model_name
            f.attrs['train_size'] = train_size
            f.attrs['test_size'] = test_size
            f.attrs['rand_seed1'] = rand_seed1
            f.attrs['rand_seed2'] = rand_seed2

        # %% Show loss evolution when training is done
        plt.figure()
        plt.plot(loss, 'o', label='loss')
        plt.plot(val_loss, '-', label='Validation loss')

        if model_structure == "Branch_Encoder_Decoder":
            loss_name_list = ['earthquake train loss', 'earthquake valid loss', 'noise train loss', 'noise valid loss']
            loss_plot_list = ['o', '', 'o', '']
            for ii in range(4):
                plt.plot(partial_loss[ii], marker=loss_plot_list[ii], label=loss_name_list[ii])

        plt.yscale('log')
        plt.legend()
        plt.title(model_name)
        plt.show()
        plt.savefig(model_dataset_dir_current + f'/{model_name}_Training_history.png')
