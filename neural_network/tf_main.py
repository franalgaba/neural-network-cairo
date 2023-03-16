import tensorflow as tf
import tensorflow_model_optimization as tfmot
import numpy as np

# Define the quantization scheme
range_min, range_max = -128, 127

# Define the network architecture
model = tf.keras.Sequential([
    tf.keras.layers.InputLayer(input_shape=(196,)),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('relu'),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Activation('softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Prepare the data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = tf.image.resize(x_train, [14,14]).numpy()
x_test = tf.image.resize(x_test, [14,14]).numpy()
x_train = x_train.reshape(-1, 196).astype(np.float32) #  / 255.0
x_test = x_test.reshape(-1, 196).astype(np.float32) # / 255.0

# Train the model
model.fit(x_train, y_train, epochs=50)

# Evaluate the model
model.evaluate(x_test, y_test)

def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(x_train).batch(1).take(100):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model_quant)

interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
interpreter.allocate_tensors()

tensor_details = interpreter.get_tensor_details()

for dict in tensor_details:
    i = dict['index']
    tensor_name = dict['name']
    scales = dict['quantization_parameters']['scales']
    zero_points = dict['quantization_parameters']['zero_points']
    tensor = interpreter.tensor(i)()

    # print(tensor)
    print(f"name: {tensor_name}")
    print(f"scale: {scales}")
    print(f"zero_point: {zero_points}")
    print("------------------------------")
    # print(i, type, tensor_name, scales.shape, zero_points.shape, tensor.shape)