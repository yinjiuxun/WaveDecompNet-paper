# %% import modules
from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras

# %% load model
model = keras.models.load_model('./Model_and_datasets/Synthetic_seismogram_Z_Conv1DTranspose_ENZ.hdf5')
# %% load dataset
with h5py.File('./Model_and_datasets/Synthetic_seismogram_Autoencoder_model_datasets_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_test = f['X_test'][:]
    Y_test = f['Y_test'][:]

# %% predict the waveforms
Y_predict = model.predict(X_test)

components = "ENZ"
# %% Visualize the results
i_show = np.random.randint(0, X_test.shape[0], 5)
plt.close('all')
fig, ax = plt.subplots(5, 3, figsize=(16, 12), num=1, sharey=True, sharex=True)
for i in range(5):
    for j in range(3):
        ax[i, j].plot(time, X_test[i_show[i], :, j], '-k', label='X test')
        ax[i, j].plot(time, Y_test[i_show[i], :, j], '-b', label='Y test')
        ax[i, j].plot(time, Y_predict[i_show[i], :, j], '-r', label='Y predicted')

        if i == 0:
            ax[i, j].set_title(components[j])
            if j == 0:
                ax[i, j].legend()
        if i == 4:
            ax[i, j].set_xlabel("Time (s)")
plt.show()
plt.savefig('./Figures/Synthetic_seismogram_Model_prediction_ENZ.png')


# %% Visualize the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file='./Figures/synthetic_seismograms_model_plot_ENZ.png', show_shapes=True, show_layer_names=True)