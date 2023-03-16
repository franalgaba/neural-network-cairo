import tensorflow as tf
import numpy as np

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) #  / 255.0

print(x_train[2])
