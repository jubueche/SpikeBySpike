from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.utils import to_categorical
import os
import numpy as np
import matplotlib.pyplot as plt

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

model = Sequential()
model.add(Dense(1000, activation='relu', input_dim = 20*500))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


one_hot_y_train = to_categorical(y_train, num_classes=10)
one_hot_y_test = to_categorical(y_test, num_classes=10)

model.fit(X_train, one_hot_y_train, epochs=100, batch_size=32, validation_data=(X_test, one_hot_y_test), shuffle=True)

"""y_hat = model.predict(X_test)
y_hat = np.argmax(y_hat, axis=1)

print(y_hat)
y_true = np.asarray(y_hat == y_test, dtype=np.int)
print(y_true)
accuracy = np.sum(y_true) / len(y_true)

print(accuracy)"""

