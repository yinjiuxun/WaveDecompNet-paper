import keras
from keras.models import Sequential
from keras.layers import Conv1D, AveragePooling1D, MaxPooling1D, UpSampling1D, LeakyReLU, Conv1DTranspose, \
    BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Conv2DTranspose
from keras.layers import Dense, Dropout, Flatten
from keras.layers import LSTM, GRU, Bidirectional
import tensorflow as tf


def autoencoder_test1(input_shape):
    model = Sequential()
    model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))
    model.add(Conv1D(8, 7, padding='same', activation='relu'))
    model.add(Conv1D(1, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model


def autoencoder_Conv1DTranspose(input_shape):
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(3))  # 25
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(32, 3, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 150
    model.add(Conv1DTranspose(16, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(Conv1DTranspose(1, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model


def autoencoder_Conv1DTranspose2(input_shape):
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))  # 37
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(2))  # 76
    model.add(Conv1D(32, 2, activation='relu'))
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(16, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(Conv1DTranspose(1, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model


def autoencoder_Conv1DTranspose_ENZ(input_shape):
    model_name = "autoencoder_Conv1DTranspose_ENZ"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(2, padding='same'))  # 37
    model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(2))  # 76
    model.add(Conv1D(32, 2, activation='relu'))
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(16, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


def autoencoder_Conv1DTranspose_ENZ2(input_shape):
    model_name = "autoencoder_25features_Conv1DTranspose_ENZ"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 7, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(16, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))  # 25
    # model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(32, 3, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(16, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


def autoencoder_Conv1DTranspose_ENZ3(input_shape):
    model_name = "autoencoder_Conv1DTranspose_ENZ_Bing"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 9, padding='same', activation='relu', input_shape=input_shape))
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling1D(3, padding='same'))  # 25
    # model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    # model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(64, 3, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(32, 5, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


def autoencoder_Conv1DTranspose_ENZ4(input_shape):
    model_name = "AE_ENZ_BatchNormalization"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 9, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, padding='same'))  # 25
    # model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    # model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(32, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


def autoencoder_Conv1DTranspose_ENZ5(input_shape):
    model_name = "AE_ENZ_BatchNormalization2"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 9, padding='same', activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(32, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, padding='same'))  # 25
    # model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=32, return_sequences=True, dropout=0.1)))
    # model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(64, 3, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(32, 5, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(BatchNormalization())
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


def autoencoder_Conv1DTranspose_ENZ6(input_shape):
    model_name = "AE_ENZ_LeakyReLU"
    # use Conv1DTranspose instead
    model = Sequential()
    model.add(Conv1D(8, 9, padding='same', input_shape=input_shape))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 300
    model.add(Conv1D(16, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 150
    model.add(Conv1D(32, 5, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(2))  # 75
    model.add(Conv1D(64, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(3, padding='same'))  # 25
    # model.add(Dropout(rate=0.1))
    model.add(Bidirectional(LSTM(units=16, return_sequences=True, dropout=0.1)))
    # model.add(LSTM(units=16, return_sequences=True))
    model.add(UpSampling1D(3))  # 75
    model.add(Conv1DTranspose(64, 3, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 152
    model.add(Conv1DTranspose(32, 5, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 300
    model.add(Conv1DTranspose(16, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(UpSampling1D(2))  # 600
    model.add(Conv1DTranspose(8, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))
    model.add(BatchNormalization())
    model.add(Conv1DTranspose(3, 7, padding='same'))
    model.add(LeakyReLU(alpha=0.5))

    return model, model_name


