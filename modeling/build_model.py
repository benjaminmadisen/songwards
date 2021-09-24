import tensorflow as tf
import numpy as np

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(21,)),
        tf.keras.layers.Dense(32, activation="relu", name="layer1"),
        tf.keras.layers.Dense(32, activation="relu", name="layer2"),
        tf.keras.layers.Dense(2, activation="softmax", name="layer3"),
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

x = np.random.random((32,21))
y = np.random.randint(2, size=32)
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

model.fit(ds, epochs=5)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open('modeling/model.tflite', 'wb') as f:
  f.write(tflite_model)