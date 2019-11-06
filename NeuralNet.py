from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
import keras
from keras.utils import to_categorical
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

try:
    file_path = os.path.join(os.getcwd(), "DYNAPS/Resources/Simulation/Dataset")
    X_train = np.load(os.path.join(file_path, "Training_spikes.dat"), allow_pickle=True)
    X_test = np.load(os.path.join(file_path, "Testing_spikes.dat"), allow_pickle=True)
    y_test = np.load(os.path.join(file_path, "Testing_labels.dat"), allow_pickle=True)
    y_train = np.load(os.path.join(file_path, "Training_labels.dat"), allow_pickle=True)
    
except:
    raise Exception("Failed to load data. Please run $ python main.py -audio")

# Reshape
n_train = X_train.shape[0]
n_test = X_test.shape[0]
X_train = np.reshape(X_train, (n_train,-1))
X_test =  np.reshape(X_test, (n_test,-1))

frequencies,times,spectrogram = signal.spectrogram(X_train[0], fs=8000)
plt.pcolormesh(times, frequencies, spectrogram)
plt.imshow(spectrogram)
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()