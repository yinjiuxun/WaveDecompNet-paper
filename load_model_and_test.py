# %% import modules
from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras

# %% load model
model2 = keras.models.load_model('./Ricker_Autoencoder_model.hdf5')
# %% load dataset
with h5py.File('./Ricker_Autoencoder_model_datasets.hdf5', 'r') as f:
    time = f['time'][:]
    X_test = f['X_test'][:]
    Y_test = f['Y_test'][:]

# %% predict the waveforms
Y_predict = model2.predict(X_test)

# %% Visualize the results
i = np.random.randint(0, X_test.shape[0])
plt.close('all')
plt.figure()
plt.plot(time, X_test[i,:,0], '-k', label='X test')
plt.plot(time, Y_test[i,:,0], '-b', label='Y test')
plt.plot(time, Y_predict[i,:,0], '-r', label='Y predicted')
plt.legend()
plt.show()
