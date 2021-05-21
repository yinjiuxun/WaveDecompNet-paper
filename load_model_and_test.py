# %% import modules
from matplotlib import pyplot as plt
import numpy as np
import h5py
import keras
from sklearn.model_selection import train_test_split

# %% Need to specify model_name first
model_name = 'autoencoder_Conv1DTranspose_ENZ'

# %% load model
model = keras.models.load_model('./Model_and_datasets/' + f'{model_name}_Model.hdf5')
# %% load dataset
with h5py.File('./Model_and_datasets/processed_synthetic_datasets_ENZ.hdf5', 'r') as f:
    time = f['time'][:]
    X_train = f['X_train'][:]
    Y_train = f['Y_train'][:]

# split the model based on the information provided by the model
with h5py.File('./Model_and_datasets/' + f'/{model_name}_Dataset_split.hdf5', 'r') as f:
    train_size = f.attrs['train_size']
    rand_seed1 = f.attrs['rand_seed1']
    rand_seed2 = f.attrs['rand_seed2']

X_train, X_test, Y_train, Y_test = train_test_split(X_train, Y_train, train_size=0.6, random_state=rand_seed1)
X_validate, X_test, Y_validate, Y_test = train_test_split(X_test, Y_test, test_size=0.5, random_state=rand_seed2)

# %% predict the waveforms
Y_predict = model.predict(X_test)

components = "ENZ"
# %% Visualize the results
i_show = np.random.randint(0, X_test.shape[0], 5)
plt.close('all')
fig, ax = plt.subplots(5, 3, figsize=(10, 8), num=1, sharey=True, sharex=True)
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
plt.savefig(f'./Figures/{model_name}_Prediction.png')


# %% Visualize the model
from keras.utils.vis_utils import plot_model
plot_model(model, to_file=f'./Figures/{model_name}_Visual_model.png', show_shapes=True, show_layer_names=True)
