import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype(np.int8) #  / 255.0

example_image = x_train[0]
example_label = y_train[0]

print(np.array2string(x_train[0], separator=','))

print("writing sample image...")
with open("image.txt", "w") as f:
    for weight in example_image.flatten().tolist():
        if weight > 0:
            sign = "false"
        else:
            sign = "true"
        addition = "tensor.append(i33 [ inner: {}_u32, sign: {} ]);\n".format(str(int(abs(weight))), sign)
        addition = addition.replace("[", "{")
        addition = addition.replace("]", "}")
        f.write(addition)