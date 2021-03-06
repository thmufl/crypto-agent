
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

inputs = layers.Dense(shape=(4,), name="digits")
x = layers.Dense(64, activation="relu", name="dense_1")(inputs)
x = layers.Dense(64, activation="relu", name="dense_2")(x)
outputs = layers.Dense(10, activation="softmax", name="predictions")(x)

# model = keras.Model(inputs=inputs, outputs=outputs)
