import tensorflow as tf
import tensorflowjs as tfjs
import numpy as np
import pickle
import os
import yaml
from google.cloud import storage

model = tf.keras.Sequential(
    [
        tf.keras.layers.Input(shape=(23,)),
        tf.keras.layers.Dense(32, activation="relu", name="layer1"),
        tf.keras.layers.Dense(32, activation="relu", name="layer2"),
        tf.keras.layers.Dense(2, activation="softmax", name="layer3"),
    ]
)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

x = np.random.random((32,23))
y = np.random.randint(2, size=32)
ds = tf.data.Dataset.from_tensor_slices((x, y)).batch(8)

model.fit(ds, epochs=5)
model.fit(ds, epochs=5)
tfjs.converters.save_keras_model(model, "modeling/test_tfjs/")

storage_client = storage.Client()
bucket = storage_client.bucket('songwards.appspot.com')

output_path = "modeling/test_tfjs/"
if not os.path.isdir(output_path):
    os.makedirs(output_path)
with open(".songwards_config", 'r') as config_file:
    config_vars = yaml.load(config_file, Loader=yaml.CLoader)
model_outputs = os.listdir(output_path)
for model_output in model_outputs:
    if model_output in config_vars['valid_tfjs_paths']:
        blob = bucket.blob(config_vars['valid_tfjs_paths'][model_output])
        blob.upload_from_filename(output_path+model_output)
    else:
        print(model_output)
    os.remove(output_path+model_output)
os.rmdir(output_path)




wordvecs = {}
for s in ['country','test','fake']:
    wordvecs[s] = np.random.random((10,))
blob = bucket.blob('wordvecs')
blob.upload_from_string(pickle.dumps(wordvecs))