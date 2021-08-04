import keras
from keras.models import Sequential, Model
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose, \
    BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose, Add, Input
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, Bidirectional
import tensorflow as tf


import torch
from torch import nn
import torch.nn.functional as F

class Autoencoder_Conv1D_deep(nn.Module):
    def __init__(self): #TODO: FIGURE OUT THE PADDING AND STRIDE!
        super(Autoencoder_Conv1D_deep, self).__init__()
        self.enc1 = nn.Conv1d(3, 8, 9, padding='same', dtype=torch.float64)
        self.enc2 = nn.Conv1d(8, 8, 9, stride=2, padding=4, dtype=torch.float64)
        self.enc3c = nn.Conv1d(8, 16, 7, padding='same', dtype=torch.float64)
        self.enc4 = nn.Conv1d(16, 16, 7, stride=2, padding=3, dtype=torch.float64)
        self.enc5c = nn.Conv1d(16, 32, 5, padding='same', dtype=torch.float64)
        self.enc6 = nn.Conv1d(32, 32, 5, stride=2, padding=2, dtype=torch.float64)
        self.enc7c = nn.Conv1d(32, 64, 3, padding='same', dtype=torch.float64)
        self.enc8 = nn.Conv1d(64, 64, 3, stride=3, dtype=torch.float64)
        # Consider the bottleneck here
        self.dec1c = nn.ConvTranspose1d(64, 64, 3, stride=3, dtype=torch.float64)
        self.dec2 = nn.ConvTranspose1d(64, 32, 3, padding=1, dtype=torch.float64)
        self.dec3c = nn.ConvTranspose1d(32, 32, 5, stride=2, padding=2, output_padding=1, dtype=torch.float64)
        self.dec4 = nn.ConvTranspose1d(32, 16, 5, padding=2, dtype=torch.float64)
        self.dec5c = nn.ConvTranspose1d(16, 16, 7, stride=2, padding=3, output_padding=1, dtype=torch.float64)
        self.dec6 = nn.ConvTranspose1d(16, 8, 7, padding=3, dtype=torch.float64)
        self.dec7 = nn.ConvTranspose1d(8, 8, 9, stride=2, padding=4, output_padding=1, dtype=torch.float64)
        self.dec8 = nn.ConvTranspose1d(8, 3, 9, padding=4, dtype=torch.float64)

    def batch_normalize(self, num_features, x):
        temp = nn.BatchNorm1d(num_features, dtype=torch.float64)
        return temp(x)

    def forward(self, x):
        x = F.relu(self.batch_normalize(8, self.enc1(x)))
        x = F.relu(self.batch_normalize(8, self.enc2(x)))
        x1 = self.enc3c(x)
        x = F.relu(self.batch_normalize(16, x1))
        x = F.relu(self.batch_normalize(16, self.enc4(x)))
        x2 = self.enc5c(x)
        x = F.relu(self.batch_normalize(32, x2))
        x = F.relu(self.batch_normalize(32, self.enc6(x)))
        x3 = self.enc7c(x)
        x = F.relu(self.batch_normalize(64, self.enc8(x3)))

        x = self.dec1c(x)
        x = F.relu(self.batch_normalize(64, x + x3))
        x = F.relu(self.batch_normalize(32, self.dec2(x)))
        x = self.dec3c(x)
        x = F.relu(self.batch_normalize(32, x + x2))
        x = F.relu(self.batch_normalize(16, self.dec4(x)))
        x = self.dec5c(x)
        x = F.relu(self.batch_normalize(16, x + x1))
        x = F.relu(self.batch_normalize(8, self.dec6(x)))
        x = F.relu(self.batch_normalize(8, self.dec7(x)))
        x = self.batch_normalize(3, self.dec8(x))

        return x

modelx = Autoencoder_Conv1D_deep()
X2 = modelx(X_train)


def autoencoder_Conv1DTranspose_ENZ8(input_shape):
    model_name = "AE_LeakyReLU_skip_connection_deep"
    # use Conv1DTranspose instead
    input_img = Input(shape=input_shape)
    y = Conv1D(8, 9, padding='same')(input_img)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1D(8, 9, strides=2, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    #y = MaxPooling1D(2)(y)

    y1 = Conv1D(16, 7, padding='same')(y)  # skip connection 1
    y = LeakyReLU(alpha=0.5)(y1)
    y = BatchNormalization()(y)

    y = Conv1D(16, 7, strides=2, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    #y = MaxPooling1D(2)(y)

    y2 = Conv1D(32, 5, padding='same')(y)  # skip connection 2
    y = LeakyReLU(alpha=0.5)(y2)
    y = BatchNormalization()(y)

    y = Conv1D(32, 5, strides=2, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    #y = MaxPooling1D(2)(y)

    y3 = Conv1D(64, 3, padding='same')(y)  # skip connection 3
    y = LeakyReLU(alpha=0.5)(y3)
    y = BatchNormalization()(y)

    y = Conv1D(64, 3, strides=3, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    #y = MaxPooling1D(3, padding='same')(y)

    y = Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1))(y)

    y = UpSampling1D(3)(y)
    y = Conv1DTranspose(64, 3, padding='same')(y)
    y = Add()([y3, y])  # skip connection 3
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1D(64, 3, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = UpSampling1D(2)(y)

    y = Conv1DTranspose(32, 5, padding='same')(y)
    y = Add()([y2, y])  # skip connection 2
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1D(32, 5, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = UpSampling1D(2)(y)  # 300

    y = Conv1DTranspose(16, 7, padding='same')(y)
    y = Add()([y1, y])  # skip connection 1
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1D(16, 7, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = UpSampling1D(2)(y)  # 600

    y = Conv1DTranspose(8, 7, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1D(8, 7, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)
    y = BatchNormalization()(y)

    y = Conv1DTranspose(3, 7, padding='same')(y)
    y = LeakyReLU(alpha=0.5)(y)

    model = Model(input_img, y)
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model, model_name
