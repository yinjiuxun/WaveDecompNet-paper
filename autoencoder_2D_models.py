import keras
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose, \
    BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, Bidirectional
import tensorflow as tf


def autoencoder_Conv2D_Spectrogram(input_shape):
    """a model to deal with spectrogram data normalized to Min and Max"""
    model_name = "spectrogram_real_imag_MinMax"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    model.add(Conv2D(6, 1, padding='same', activation='sigmoid'))

    return model, model_name


def autoencoder_Conv2D_Spectrogram2(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_real_imag_standard"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    # model.add(BatchNormalization())
    model.add(Conv2DTranspose(6, 1, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())  # NEW

    return model, model_name


def autoencoder_Conv2D_Spectrogram3(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_mask_l1_softmax"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2DTranspose(6, 1, padding='same',
                              activity_regularizer=tf.keras.regularizers.l1(0.01),
                              activation='softmax'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model, model_name


def autoencoder_Conv2D_Spectrogram4(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_mask_softmax"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2DTranspose(6, 1, padding='same', activation='softmax'))
    # model.add(Conv2DTranspose(6, 1, padding='same',
    #                           activity_regularizer=tf.keras.regularizers.l2(0.0),
    #                           activation='softmax'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model, model_name


def autoencoder_Conv2D_Spectrogram76(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_mask_l1_sigmoid"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2DTranspose(6, 1, padding='same', activation='softmax'))
    model.add(Conv2DTranspose(6, 1, padding='same',
                              activity_regularizer=tf.keras.regularizers.l1(0.01),
                              activation='softmax'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model, model_name


def autoencoder_Conv2D_Spectrogram77(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_mask_l1_sigmoid"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    # model.add(Conv2DTranspose(6, 1, padding='same', activation='softmax'))
    model.add(Conv2DTranspose(6, 1, padding='same',
                              activity_regularizer=tf.keras.regularizers.l1(0.01),
                              activation='softmax'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model, model_name


def autoencoder_Conv2D_Spectrogram78(input_shape):
    """a model to deal with spectrogram data normalized to zero-mean and unit variance"""
    model_name = "spectrogram_mask_softmax"
    model = Sequential()
    model.add(Conv2D(16, 5, padding='same', activation='relu', input_shape=input_shape))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))  # 150
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(32, 3, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(UpSampling2D((2, 2)))  # 150
    model.add(Conv2DTranspose(16, 5, padding='same', activation='relu'))
    # model.add(BatchNormalization())
    model.add(Conv2DTranspose(6, 1, padding='same', activation='softmax'))
    model.compile(loss='mean_squared_logarithmic_error', optimizer='adam')

    return model, model_name