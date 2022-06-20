import torch
from torch.utils.data import Dataset
import os
import h5py
import numpy as np
import time


class WaveformDataset(Dataset):
    def __init__(self, X_train, Y_train):
        self.X_train = np.moveaxis(X_train, 1, -1)
        self.Y_train = np.moveaxis(Y_train, 1, -1)

    def __len__(self):
        return self.X_train.shape[0]

    def __getitem__(self, idx):
        X_waveform = self.X_train[idx]
        Y_waveform = self.Y_train[idx]
        return X_waveform, Y_waveform


class WaveformDataset_h5(Dataset):
    def __init__(self, annotations_file):
        self.hdf5_file = h5py.File(annotations_file, 'r')

    def __len__(self):
        return self.hdf5_file['X_train'].shape[0]

    def __getitem__(self, idx):
        X_waveform = self.hdf5_file['X_train'][idx]
        Y_waveform = self.hdf5_file['Y_train'][idx]
        return X_waveform, Y_waveform


# from https://github.com/Bjarten/early-stopping-pytorch
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


# PyTorch
# Try the new loss function
class Explained_Variance_Loss(torch.nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(Explained_Variance_Loss, self).__init__()

    def forward(self, inputs, targets):
        return torch.var(targets - inputs, dim=2, unbiased=True, keepdim=True) \
               / torch.var(inputs, dim=2, unbiased=True, keepdim=True)


def try_gpu(i=0):  # @save
    """Return gpu(i) if exists, otherwise return cpu()."""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


# count total number of parameters of a model
def parameter_number(model):
    num_param = 0
    for parameter in model.parameters():
        # print(parameter)
        num_param += np.prod(parameter.shape)
    return num_param


def training_loop(train_dataloader, validate_dataloader, model, loss_fn, optimizer, scheduler, epochs, patience, device):
    # to track the training loss as the model trains
    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)

    for epoch in range(1, epochs + 1):
        # estimate time for each epoch
        starttime = time.time()

        # ======================= training =======================
        # initialize the model for training
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

        if scheduler is not None: # Adjust the learning rate
            scheduler.step() 
            
        # ======================= validating =======================
        # initialize the model for training
        model.eval()
        for X, y in validate_dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)

            # record validation loss
            valid_losses.append(loss.item())

        # calculate average loss over an epoch
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # print training/validation statistics
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}\n' +
                     f'time per epoch: {(time.time() - starttime):.3f} s')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        # load the last checkpoint with the best model
    model.load_state_dict(torch.load('checkpoint.pt'))

    return model, avg_train_losses, avg_valid_losses


def training_loop_branches(train_dataloader, validate_dataloader, model, loss_fn, optimizer, scheduler, epochs
                           , patience, device, minimum_epochs=None):

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    avg_train_losses1 = []  # earthquake average loss with epoch
    avg_train_losses2 = []  # noise average loss with epoch

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []
    avg_valid_losses1 = []  # earthquake average loss with epoch
    avg_valid_losses2 = []  # noise average loss with epoch

    # initialize the early_stopping object
    if patience is None: # dont apply early stopping
        early_stopping = EarlyStopping(patience=1, verbose=False)
    else:
        early_stopping = EarlyStopping(patience=patience, verbose=True, delta=1e-6)

    for epoch in range(1, epochs + 1):
        # estimate time for each epoch
        starttime = time.time()

        # to track the training loss as the model trains
        train_losses = []
        train_losses1 = []  # earthquake loss
        train_losses2 = []  # noise loss

        # to track the validation loss as the model trains
        valid_losses = []
        valid_losses1 = []  # earthquake loss
        valid_losses2 = []  # noise loss

        # ======================= training =======================
        # initialize the model for training
        model.train()
        size = len(train_dataloader.dataset)
        for batch, (X, y) in enumerate(train_dataloader):
            # Compute prediction and loss
            X, y = X.to(device), y.to(device)
            pred1, pred2 = model(X)
            loss1 = loss_fn(pred1, y)
            loss2 = loss_fn(pred2, X - y)

            loss = loss1 + loss2

            # record training loss
            train_losses.append(loss.item())
            train_losses1.append(loss1.item())
            train_losses2.append(loss2.item())

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if scheduler is not None: # Adjust the learning rate
            scheduler.step() 

        # ======================= validating =======================
        # initialize the model for training
        model.eval()
        for X, y in validate_dataloader:
            X, y = X.to(device), y.to(device)
            pred1, pred2 = model(X)
            loss1 = loss_fn(pred1, y)
            loss2 = loss_fn(pred2, X - y)

            #loss = loss1 + loss2

            # record validation loss
            valid_losses.append(loss1.item() + loss2.item())
            valid_losses1.append(loss1.item())
            valid_losses2.append(loss2.item())

        # calculate average loss over an epoch
        # total loss
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)

        # earthquake waveform loss
        train_loss1 = np.average(train_losses1)
        valid_loss1 = np.average(valid_losses1)
        avg_train_losses1.append(train_loss1)
        avg_valid_losses1.append(valid_loss1)

        # ambient noise waveform loss
        train_loss2 = np.average(train_losses2)
        valid_loss2 = np.average(valid_losses2)
        avg_train_losses2.append(train_loss2)
        avg_valid_losses2.append(valid_loss2)

        # print training/validation statistics
        epoch_len = len(str(epochs))
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
                     f'train_loss: {train_loss:.5f} ' +
                     f'valid_loss: {valid_loss:.5f}\n' +
                     f'time per epoch: {(time.time() - starttime):.3f} s')

        print(print_msg)

        # clear lists to track next epoch
        train_losses = []
        valid_losses = []

        if patience is not None:
            if (minimum_epochs is None) or ((minimum_epochs is not None) and (epoch > minimum_epochs)):
                # early_stopping needs the validation loss to check if it has decresed,
                # and if it has, it will make a checkpoint of the current model
                early_stopping(valid_loss, model)

            if early_stopping.early_stop:
                print("Early stopping")
                break

    # load the last checkpoint with the best model if apply early stopping
    if patience is not None:
        model.load_state_dict(torch.load('checkpoint.pt'))

    partial_loss = [avg_train_losses1, avg_valid_losses1, avg_train_losses2, avg_valid_losses2]

    return model, avg_train_losses, avg_valid_losses, partial_loss


def model_same(model1, model2):
    """Function to tell if two models are the same (same parameters)"""
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
        else:
            return True