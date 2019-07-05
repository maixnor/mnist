
# script for handwritten digits recognition
# made with keras and tensorflow and the mnist dataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt 
import sys

# command line arguments for input section (-i)
arg = ""
try:
    arg = str(sys.argv[1])
except IndexError as identifier:
    pass

# loading mnist dataset and splitting into training and testing data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#norminalizing data from values 0:255 to 0:1
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

#mdoel definition

# different activation methods for different layers
activation1 = tf.nn.crelu
activation2 = tf.nn.elu
activation3 = tf.nn.sigmoid

# number of nodes in hidden layers
layer1 = 128
layer2 = 128

# optimizer for compiler
optim = "adam"

# model with 4 layers, 2 hidden laayers
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(layer1, activation = activation1))
model.add(tf.keras.layers.Dense(layer2, activation = activation2))
model.add(tf.keras.layers.Dense(10, activation = activation3))

# model gets compiled and optimized
model.compile(optimizer=optim,
              loss="sparse_categorical_crossentropy",
              metrics=['accuracy'])

# model is tested with the test data
model.fit(x_train, y_train, epochs=2)

# loss and accuracy evaluations
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_acc, val_loss)

# documentation for further analysis 
with open('log.txt', mode = 'a') as file:
    file.write(("{} {} {} {} {} {} {} {}\n".format(str(val_acc), str(val_loss.round(4)), str(layer1), str(layer2), \
        str(activation1).split(' ')[1], str(activation2).split(' ')[1], str(activation3).split(' ')[1], optim)))
    file.close()

# predictions from test data
predictions = model.predict([x_test])

# input section for easier demonstration and verification
if arg == "-i":
    while(True):
        idx = input('which index would you like to see?\n')

        if idx == '':
            break

        idx = int(idx)

        print(np.argmax(predictions[idx]))

        plt.imshow(x_test[idx], cmap = plt.cm.binary)
        plt.show()
