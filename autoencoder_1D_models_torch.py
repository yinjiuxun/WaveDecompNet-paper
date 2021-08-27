# import keras
# from keras.models import Sequential, Model
# from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose, \
#     BatchNormalization
# from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose, Add, Input
# from keras.layers import Dense, Dropout, Flatten
# from keras.layers import LSTM, GRU, Bidirectional
# import tensorflow as tf


import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder_Conv1D_deep(nn.Module):
    def __init__(self, model_name): #TODO: FIGURE OUT THE PADDING AND STRIDE!
        super(Autoencoder_Conv1D_deep, self).__init__()
        self.model_name = model_name
        self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        self.enc8 = nn.Conv1d(64, 64, 3, stride=3, dtype=torch.float64)

        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)

        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn13 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn14 = nn.BatchNorm1d(8, dtype=torch.float64)

        self.bn3 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(16, dtype=torch.float64)

        self.bn5 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn9 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn10 = nn.BatchNorm1d(32, dtype=torch.float64)

        self.bn7 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.bn8 = nn.BatchNorm1d(64, dtype=torch.float64)

        self.bn15 = nn.BatchNorm1d(3, dtype=torch.float64)


    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.bn3(x1))
        x = F.relu(self.bn4(self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.bn5(x2))
        x = F.relu(self.bn6(self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.bn7(self.enc8(x3)))

        x = self.dec1c(x)
        x = F.relu(self.bn8(x + x3))
        x = F.relu(self.bn9(self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.bn10(x + x2))
        x = F.relu(self.bn11(self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.bn12(x + x1))
        x = F.relu(self.bn13(self.dec6(x)))
        x = F.relu(self.bn14(self.dec7(x)))
        x = self.bn15(self.dec8(x))

        return x


class Autoencoder_Conv1D_deep_LSTM(nn.Module):
    """Add the 2-layer LSTM layer as the bottleneck"""
    def __init__(self, model_name): #TODO: FIGURE OUT THE PADDING AND STRIDE!
        super(Autoencoder_Conv1D_deep_LSTM, self).__init__()
        self.model_name = model_name
        self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        self.enc8 = nn.Conv1d(64, 64, 3, stride=3, dtype=torch.float64)
        # # Consider the bottleneck here, size of enc8 output:[128, 64, 25]
        self.bottleneck = nn.LSTM(64, 32, 2, bidirectional=True,
                                  batch_first=True, dtype=torch.float64)

        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)

        self.bn1 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn2 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn13 = nn.BatchNorm1d(8, dtype=torch.float64)
        self.bn14 = nn.BatchNorm1d(8, dtype=torch.float64)

        self.bn3 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn4 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn11 = nn.BatchNorm1d(16, dtype=torch.float64)
        self.bn12 = nn.BatchNorm1d(16, dtype=torch.float64)

        self.bn5 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn6 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn9 = nn.BatchNorm1d(32, dtype=torch.float64)
        self.bn10 = nn.BatchNorm1d(32, dtype=torch.float64)

        self.bn7 = nn.BatchNorm1d(64, dtype=torch.float64)
        self.bn8 = nn.BatchNorm1d(64, dtype=torch.float64)

        self.bn15 = nn.BatchNorm1d(3, dtype=torch.float64)


    def forward(self, x):
        x = F.relu(self.bn1(self.enc1(x)))
        x = F.relu(self.bn2(self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.bn3(x1))
        x = F.relu(self.bn4(self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.bn5(x2))
        x = F.relu(self.bn6(self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.bn7(self.enc8(x3)))

        #print(x.shape)
        x = x.permute(0, 2, 1)
        #print(x.shape)
        x, _ = self.bottleneck(x)
        x = x.permute(0, 2, 1)
        #print(x.shape)

        x = self.dec1c(x)
        x = F.relu(self.bn8(x + x3))
        x = F.relu(self.bn9(self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.bn10(x + x2))
        x = F.relu(self.bn11(self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.bn12(x + x1))
        x = F.relu(self.bn13(self.dec6(x)))
        x = F.relu(self.bn14(self.dec7(x)))
        x = self.bn15(self.dec8(x))

        return x