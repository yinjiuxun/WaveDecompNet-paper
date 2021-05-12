# %% import modules
from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras

# %% load model
model = keras.models.load_model('./Model_and_datasets/Synthetic_seismogram_Z_Conv1DTranspose_38.hdf5')
# %% load dataset
with h5py.File('./Model_and_datasets/Synthetic_seismogram_Z_Autoencoder_model_datasets.hdf5', 'r') as f:
    time = f['time'][:]
    X_test = f['X_test'][:]
    Y_test = f['Y_test'][:]

# %% predict the waveforms
Y_predict = model.predict(X_test)

# %% Visualize the results
i_show = np.random.randint(0, X_test.shape[0], 10)
plt.close('all')
fig, ax = plt.subplots(5, 2, figsize=(16, 12), squeeze=True, num=1, sharey=True, sharex=True)
ax = ax.flatten()
for i, axi in enumerate(ax):
    axi.plot(time, X_test[i_show[i], :, 0], '-k', label='X test')
    axi.plot(time, Y_test[i_show[i], :, 0], '-b', label='Y test')
    axi.plot(time, Y_predict[i_show[i], :, 0], '-r', label='Y predicted')
    if i == 0:
        axi.legend()
ax[-2].set_xlabel('Time (s)')
ax[-1].set_xlabel('Time (s)')
plt.show()
plt.savefig('./Figures/Synthetic_seismogram_Z_Model_prediction.png')


# %% Visualize the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./Figures/synthetic_seismograms_Z_model_plot.png', show_shapes=True, show_layer_names=True)