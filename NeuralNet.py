from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Conv2D, BatchNormalization, MaxPooling2D, Flatten
from keras.optimizers import Adam
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

"""model = Sequential()

model.add(Dense(64, activation='relu', input_dim=500))
model.add(Dropout(0.7))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.7))
model.add(Dense(10, activation='softmax'))

adam = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy',
              optimizer=adam,
              metrics=['accuracy'])

model.fit(X_train, one_hot_y_train,
          epochs=3,
          batch_size=128, validation_data=(X_test,one_hot_y_test))
"""

# Shuffle X_train and X_test
train_indices = np.linspace(0,n_train-1,n_train).astype(np.int)
np.random.shuffle(train_indices)
test_indices = np.linspace(0,n_test-1,n_test).astype(np.int)
np.random.shuffle(test_indices)

X_train = X_train[train_indices,:]
y_train = y_train[train_indices]
y_test = y_test[test_indices]
X_test = X_test[test_indices,:]

one_hot_y_train = to_categorical(y_train, num_classes=10)
one_hot_y_test = to_categorical(y_test, num_classes=10)

current_idx = 0
class Voice:

    def __init__(self, X_train, y_train, batch_size):
        self.X = X_train
        self.y = y_train
        self.curr_idx = 0
        self.batch_size = batch_size
        self.n_train = X_train.shape[0]

    def next_batch(self):
        if(self.curr_idx + self.batch_size < self.n_train):
            self.curr_idx = 0
        data = (self.X[self.curr_idx:self.curr_idx+self.batch_size,:], self.y[self.curr_idx:self.curr_idx+self.batch_size,:])
        self.curr_idx += self.batch_size
        return data


import tensorflow as tf
from tensorflow.contrib import rnn 

learning_rate = 0.02
training_steps = 5000
batch_size = 1600
display_step = 250

voice = Voice(X_train, one_hot_y_train, batch_size)

# Network Parameters
num_input = 1
timesteps = 500 # timesteps
num_hidden = 64 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

def RNN(x):
    # Define a lstm cell with tensorflow
    lstm_cell = rnn.LSTMBlockCell(
        num_hidden, forget_bias=1.0)

    # Get lstm cell output
    # outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)
    outputs, states = tf.nn.dynamic_rnn(
        cell=lstm_cell, inputs=x, time_major=False, dtype=tf.float32)
    
    output_layer = tf.layers.Dense(
        num_classes, activation=None, 
        kernel_initializer=tf.orthogonal_initializer()
    )
    return output_layer(tf.layers.batch_normalization(outputs[:, -1, :]))


# Need to clear the default graph before moving forward
tf.reset_default_graph()
graph = tf.Graph()
with graph.as_default():
    tf.set_random_seed(1)
    # tf Graph input
    X = tf.placeholder("float", [None, timesteps, num_input])
    Y = tf.placeholder("float", [None, num_classes])
    logits = RNN(X)
    prediction = tf.nn.softmax(logits)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=logits, labels=Y))
    optimizer = tf.train.AdamOptimizer() # Use Adam
    gvs = optimizer.compute_gradients(loss_op)
    capped_gvs = [
        (tf.clip_by_norm(grad, 2.), var) if not var.name.startswith("dense") else (grad, var)
        for grad, var in gvs]
    for _, var in gvs:
        if var.name.startswith("dense"):
            print(var.name)    
    train_op = optimizer.apply_gradients(capped_gvs)  

    # Evaluate model (with test logits, for dropout to be disabled)
    correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    
    print("All parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.global_variables()]))
    print("Trainable parameters:", np.sum([np.product([xi.value for xi in x.get_shape()]) for x in tf.trainable_variables()]))

config = tf.ConfigProto()
best_val_acc = 0.0
with tf.Session(graph=graph, config=config) as sess:
    # Run the initializer
    sess.run(init)
    for step in range(1, training_steps+1):
        batch_x, batch_y = voice.next_batch()
        # Reshape data to get 28 seq of 28 elements
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            # Calculate accuracy for 128 mnist test images
            test_len = 128
            test_data = X_test[:test_len,:].reshape((-1, timesteps, num_input))
            test_label = one_hot_y_test[:test_len,:]
            val_acc = sess.run(accuracy, feed_dict={X: test_data, Y: test_label})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc) + ", Test Accuracy= " + \
                  "{:.3f}".format(val_acc))
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                save_path = saver.save(sess, "/tmp/model.ckpt", global_step=step)
                print("Model saved in path: %s" % save_path)
    print("Optimization Finished!")