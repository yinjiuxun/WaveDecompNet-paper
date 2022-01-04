# WaveDecompNet
A machine learning tool to separate earthquake and ambient noise signals for the seismic data in time domain.

\Add the network figure

Step 1: Download the event catalog based on distance range to a given station 


Step 2: 

2a - Download contineous seismic data

2b - Get the STEAD earthquake waveform data with SNR > 40dB and stack with STEAD noise signals

2c - Shuffle the phase of contineous data from a given station to get the local ambient noise signals, then stack with 
the STEAD earthquake waveform.

2d - Combine the waveforms datasets from 2b and 2c to get the final datasets that will be used to train the 
WaveDecompNet.


Step 3: Training the WaveDecompNet using the prepared datasets.


Step 4: autoencoder_denoise_test.py


Step 5: load_model_and_test.py