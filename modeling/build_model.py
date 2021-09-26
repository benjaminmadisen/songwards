
from gensim.models import KeyedVectors, Word2Vec
import numpy as np
import re
import pickle
import os
import yaml
import requests
from tqdm import tqdm
import requests
import sqlite3
from urllib.request import urlretrieve
from gensim import utils
from google.cloud import secretmanager, storage
from main import make_spotify_request



class SpotifyDbCorpus:
    def __init__(self, db_path, max_rows = 100):
        self.con = sqlite3.connect(db_path)
        self.max_rows = max_rows

    def __iter__(self):
        cur = self.con.cursor()
        its = 0
        for row in cur.execute("SELECT A.track_id, C.name FROM track_playlist1 A JOIN playlist C ON A.playlist_id = C.id"):
            its += 1
            if its > self.max_rows:
                break
            sentence = ("%s %s" % (row[0], row[1])).split(" ")
            yield sentence

with open(".songwards_config", 'r') as config_file:
    config_vars = yaml.load(config_file, Loader=yaml.CLoader)
storage_client = storage.Client()
bucket = storage_client.bucket(config_vars['bucket_path'])

def write_saved_model_to_gcloud(model_path, valid_tfjs_paths):
    model_outputs = os.listdir(model_path)
    for model_output in model_outputs:
        if model_output in valid_tfjs_paths:
            blob = bucket.blob(valid_tfjs_paths[model_output])
            blob.upload_from_filename(model_path+model_output)
        else:
            print(model_output)
        os.remove(model_path+model_output)
    os.rmdir(model_path)

def write_saved_wordvecs_to_gcloud(wordvec_path, saved_path):
    word_vecs = KeyedVectors.load(wordvec_path, mmap='r')
    useful_words = [itk for itk in word_vecs.index_to_key if not re.search(r'\d', itk)]
    wordvecs = {}
    for word in useful_words:
        wordvecs[word] = word_vecs[word]
    blob = bucket.blob(saved_path)
    blob.upload_from_string(pickle.dumps(wordvecs))

def create_raw_data_database(raw_data_url, db_path, db_name):
    urlretrieve(raw_data_url, db_path+"temp.sql")
    create_database_from_file(db_path, db_name)

def create_database_from_file(db_path, db_name):
    con = sqlite3.connect(db_path+db_name) 
    cur = con.cursor()

    clause = ""
    with open(db_path+"temp.sql","r", encoding='utf8') as sql_file:
        for line in sql_file:
            if "  KEY" not in line:
                if "CONSTRAINT" in line:
                    line = line[line.find("FOREIGN"):].replace(" ON DELETE NO ACTION ON UPDATE NO ACTION","").replace("FOREIGN KEY ","FOREIGN KEY")
                if "UNIQUE KEY `unique_index`" not in line:
                    clause += line
                if ";" in clause:
                    if "LOCK TABLES" not in clause:
                        try:
                            cur.execute(clause.replace("ENGINE=InnoDB DEFAULT CHARSET=latin1","").replace("\\\\","").replace("\\'","").replace('\\"',"").replace(",\n)",")"))
                            con.commit()
                        except:
                            with open(db_path+"debug", "w") as handle:
                                handle.write(clause.replace("ENGINE=InnoDB DEFAULT CHARSET=latin1","").replace("\\\\","").replace("\\'","").replace('\\"',"").replace(",\n)",")"))
                                raise
                    clause = ""
    os.remove(db_path+"temp.sql")

def generate_wordvectors_from_db(db_path, max_examples=100):
    sentences = SpotifyDbCorpus(db_path, max_examples)
    model = Word2Vec(sentences=sentences)

def load_song_attributes_from_spotify(db_path, n_tracks=1000):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    af_list = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo','duration_ms','time_signature']
    cur.execute("DROP TABLE IF EXISTS `track_attributes`;")
    cur.execute("""
        
        CREATE TABLE `track_attributes` (
            `track_id` varchar(100) DEFAULT NULL,
            `danceability` double DEFAULT NULL,
            `energy` double DEFAULT NULL,
            `key` int(4) DEFAULT NULL,
            `loudness` double DEFAULT NULL,
            `mode` int(4) DEFAULT NULL,
            `speechiness` double DEFAULT NULL,
            `acousticness` double DEFAULT NULL,
            `instrumentalness` double DEFAULT NULL,
            `liveness` double DEFAULT NULL,
            `valence` double DEFAULT NULL,
            `tempo` double DEFAULT NULL,
            `duration_ms` int(10) DEFAULT NULL,
            `time_signature` int(4) DEFAULT NULL,
            FOREIGN KEY (`track_id`) REFERENCES `track` (`id`)
        );
        """)
    con.commit()
    cur.execute("""SELECT A.track_id
                     FROM track_playlist1 A
                    GROUP BY A.track_id
                    ORDER BY COUNT(*) DESC LIMIT %i""" % n_tracks)
    done = False
    ids = cur.fetchall()
    n_ids = 100
    cur_ix = 0
    while not done:
        if len(ids)-cur_ix <= n_ids:
            done = True
            n_ids = len(ids)-n_ids
        uris = ids[cur_ix:cur_ix+n_ids]
        uris = [uri[0] for uri in uris]
        cur_ix += n_ids
        try:
            print(cur_ix)
            audio_features = make_spotify_request('audio-features',{'ids':",".join(uris)})['audio_features']
            insert_string = "INSERT INTO track_attributes VALUES "+", ".join(["('"+"','".join([str(audio_features[afix][fname]) for fname in af_list])+"')" for afix in range(len(audio_features)) if audio_features[afix] is not None])+";"
            cur.execute(insert_string)
            con.commit()
        except:
            pass