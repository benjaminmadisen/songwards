
from gensim.models import KeyedVectors
import numpy as np
import re
import pickle
import os
import yaml
from google.cloud import storage

storage_client = storage.Client()
bucket = storage_client.bucket('songwards.appspot.com')

output_path = "../gparty3/data/private/models/test_tfjs/"
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




word_vecs = KeyedVectors.load("../gparty3/data/private/models/word2vec.wordvectors", mmap='r')
useful_words = [itk for itk in word_vecs.index_to_key if not re.search(r'\d', itk)]
wordvecs = {}
for word in useful_words:
    wordvecs[word] = word_vecs[word]
blob = bucket.blob('wordvecs')
blob.upload_from_string(pickle.dumps(wordvecs))