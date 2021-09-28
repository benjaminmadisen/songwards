
from gensim.models import KeyedVectors, Word2Vec
import numpy as np
import re
import pickle
import os
import yaml
import requests
import tensorflow as tf
import tensorflowjs as tfjs
from tqdm import tqdm
import requests
import sqlite3
from urllib.request import urlretrieve
from gensim import utils
from google.cloud import secretmanager, storage
from main import make_spotify_request



class SpotifyDbCorpus:
    def __init__(self, db_path, temp_file):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        self.temp_file = temp_file
        with open(self.temp_file, "w") as tf:
            self.cur.execute("DROP TABLE IF EXISTS temp_track_playlist")
            self.cur.execute("CREATE TABLE temp_track_playlist AS SELECT A.track_id, playlist_id FROM track_playlist1 A JOIN track_attributes B ON A.track_id = B.track_id ORDER BY RANDOM() LIMIT 10000000")
            for row in self.cur.execute("SELECT A.track_id, C.name FROM temp_track_playlist A JOIN playlist C ON A.playlist_id = C.id"):
                tf.write(" ".join([row[0]] + utils.simple_preprocess(row[1], deacc=True))+" \n")

    def __iter__(self):
        with open(self.temp_file, "r") as tf:
            for row in tf.readlines():
                sentence = row.split(" ")
                yield sentence

class SpotifyDbGenreCorpus:
    def __init__(self, db_path, temp_file):
        self.con = sqlite3.connect(db_path)
        self.cur = self.con.cursor()
        self.temp_file = temp_file
        with open(self.temp_file, "w") as tf:
            self.cur.execute("DROP TABLE IF EXISTS temp_track_playlist")
            self.cur.execute("CREATE TABLE temp_track_playlist AS SELECT B.genres, playlist_id FROM track_playlist1 A JOIN track_attributes B ON A.track_id = B.track_id ORDER BY RANDOM() LIMIT 100000")
            cur_genres = []
            cur_track_id = -1
            for row in self.cur.execute("SELECT playlist_id, genres FROM temp_track_playlist ORDER BY playlist_id"):
                if row[0] != cur_track_id:
                    tf.write(" ".join(cur_genres)+"\n")
                    cur_genres = []
                for genre in row[1].split(" "):
                    cur_genres.append(genre)

    def __iter__(self):
        with open(self.temp_file, "r") as tf:
            for row in tf.readlines():
                sentence = row.replace("\n","").split(" ")
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
    word_vecs = Word2Vec.load(wordvec_path, mmap='r').wv
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

def generate_wordvectors_from_db(db_path, db_name, out_name):
    sentences = SpotifyDbCorpus(db_path+db_name, db_path+"temp_file")
    model = Word2Vec(sentences=sentences, vector_size=10, min_count=10)
    os.remove(db_path+"temp_file")
    model.save(db_path+out_name)
    return model

def generate_genrevectors_from_db(db_path, db_name, out_name):
    sentences = SpotifyDbGenreCorpus(db_path+db_name, db_path+"temp_file")
    model = Word2Vec(sentences=sentences, vector_size=5, min_count=10)
    os.remove(db_path+"temp_file")
    model.save(db_path+out_name)
    return model

def generate_vector_training_data(db_path, db_name, wordvectors_name):
    model = Word2Vec.load(db_path+wordvectors_name)
    con = sqlite3.connect(db_path+db_name)
    cur = con.cursor()
    af_list = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
    sel_string = ", ".join([af_list[ix+1] for ix in range(len(af_list)-1)])
    audio_features_mins = np.array([config_vars['audio_features'][af]['min'] for af in af_list[1:]])
    audio_features_maxs = np.array([config_vars['audio_features'][af]['max'] for af in af_list[1:]])
    targets = np.zeros((1,len(model.wv['rock'])))
    af_vecs = np.zeros((1,len(af_list)-1))
    for record in cur.execute("SELECT track_id, %s FROM track_attributes" % sel_string):
        if record[0] in model.wv.index_to_key:
            target = np.array([model.wv[record[0]]])
            af_vec = np.array([[record[ix+1] for ix in range(len(af_list)-1)]])
            af_vec = (af_vec-audio_features_mins)/(audio_features_maxs-audio_features_mins)
            targets = np.concatenate([targets, target])
            af_vecs = np.concatenate([af_vecs, af_vec])
    targets = targets[1:,:]
    af_vecs = af_vecs[1:,:]
    with open(db_path+"targets.npy", "wb") as f:
        np.save(f, targets)
    with open(db_path+"af_vecs.npy", "wb") as f:
        np.save(f, af_vecs)

def generate_match_training_data(db_path, db_name, wordvectors_name, genrevectors_name, alts_per_record):
    model = Word2Vec.load(db_path+wordvectors_name)
    genre_model = Word2Vec.load(db_path+genrevectors_name)
    probs = []
    summa = 0
    for phr in model.wv.index_to_key:
        probs.append(model.wv.get_vecattr(phr, 'count'))
        summa += model.wv.get_vecattr(phr, 'count')
    probs = np.array(probs)/float(summa)
    con = sqlite3.connect(db_path+db_name)
    cur = con.cursor()
    af_list = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
    sel_string = ", ".join([af_list[ix+1] for ix in range(len(af_list)-1)])
    audio_features_mins = np.array([config_vars['audio_features'][af]['min'] for af in af_list[1:]])
    audio_features_maxs = np.array([config_vars['audio_features'][af]['max'] for af in af_list[1:]])
    trains = np.zeros((1,len(model.wv['rock'])+len(genre_model.wv['pop'])+len(af_list)+1))
    targets = np.zeros((1))
    cur.execute("DROP TABLE IF EXISTS temp_track_playlist")
    cur.execute("CREATE TABLE temp_track_playlist AS SELECT B.track_id, %s, genres, playlist_id FROM track_playlist1 A JOIN track_attributes B ON A.track_id = B.track_id ORDER BY RANDOM() LIMIT 250000" % sel_string)
    rix = 0
    for row in cur.execute("SELECT A.*, B.name, C.name FROM temp_track_playlist A JOIN track B ON A.track_id = B.id JOIN playlist C ON A.playlist_id = C.id"):
        rix += 1
        rowwords = utils.simple_preprocess(row[-1], deacc=True)
        af_vec = np.array([row[ix+1] for ix in range(len(af_list)-1)])
        af_vec = (af_vec-audio_features_mins)/(audio_features_maxs-audio_features_mins)
        af_vec = np.array([af_vec])
        genres = row[-4]
        genre_vec = np.mean(np.array([genre_model.wv[genre] for genre in genres.split(" ") if genre in genre_model.wv.index_to_key]), axis=0)
        if np.any(np.isnan(genre_vec)):
            genre_vec = np.zeros(5)
        genre_vec = 0.5+(np.array([genre_vec])/20.0)
        for word in rowwords:
            if word in model.wv.index_to_key:
                word_vecs = 0.5+(np.array([model.wv[word]])/20.0)
                for _ in range(alts_per_record):
                    r = np.random.random()
                    test_val = 0.0
                    ix = 0
                    for ix in range(len(model.wv.index_to_key)):
                        test_val += probs[ix]
                        if test_val > r:
                            break
                        ix += 1
                    word_vecs = np.concatenate([word_vecs, 0.5+(np.array([model.wv[model.wv.index_to_key[ix]]])/20.0)])
                af_vecs = np.concatenate([af_vec for _ in range(alts_per_record+1)])
                genre_vecs = np.concatenate([genre_vec for _ in range(alts_per_record+1)])
                match_array = [[0,0]]
                if word in row[-2]:
                    match_array = [[1,0]]
                match_array = match_array+[[0,0] for _ in range(alts_per_record)]
                match_vecs = np.array(match_array)
                train = np.concatenate([word_vecs, genre_vecs, af_vecs, match_vecs],axis=1)
                targs = np.array([1]+[0 for _ in range(alts_per_record)])
                
                trains = np.concatenate([trains, train])
                targets = np.concatenate([targets, targs])
        if rix % 5000 == 0:
            print(rix)
            targets = targets[1:]
            trains = trains[1:,:]
            with open(db_path+"bin_targets_%i.npy" % int(rix/5000), "wb") as f:
                np.save(f, targets)
            with open(db_path+"bin_trains_%i.npy" % int(rix/5000), "wb") as f:
                np.save(f, trains)
            trains = np.zeros((1,len(model.wv['rock'])+len(genre_model.wv['pop'])+len(af_list)+1))
            targets = np.zeros((1))

def generate_tf_dataset_from_files(db_path):
    ix = 1
    db_path = str(db_path)[2:-1]
    while os.path.exists("%sbin_targets_%i.npy" % (db_path, ix)):
        with open("%sbin_targets_%i.npy" % (db_path, ix), "rb") as f:
            targets = np.load(f)
        with open("%sbin_trains_%i.npy" % (db_path, ix), "rb") as f:
            af_vecs = np.load(f)
        for rix in range(targets.shape[0]-1):
            yield af_vecs[rix,:], [targets[rix]]
        ix += 1

def generate_tf_dataset(db_path):
    return tf.data.Dataset.from_generator(
            generate_tf_dataset_from_files,
            args=[db_path],
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape((28,)), tf.TensorShape((1,))))

def train_simple_model(db_path, save_path=None):
    dataset = generate_tf_dataset(db_path).batch(32).shuffle(buffer_size=1000)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(28,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(2, activation='softmax')])
    model.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    model.fit(dataset, epochs=5)
    if save_path is not None:
        tfjs.converters.save_keras_model(model, db_path+save_path)
    return model

def load_data_train_save_model(db_path, db_name, wordvectors_name, alts_per_record, save_path):
    generate_match_training_data(db_path, db_name, wordvectors_name, alts_per_record)
    return train_simple_model(db_path, save_path)

    
def evaluate_word(model, db_path, db_name, wordvectors_name, genrevectors_name, word, uri):
    wvmodel = Word2Vec.load(db_path+wordvectors_name)
    genre_model = Word2Vec.load(db_path+genrevectors_name)
    if word in wvmodel.wv.index_to_key:
        word_vec = np.array([0.5+(wvmodel.wv[word]/20.0)])
        print(word_vec)
        con = sqlite3.connect(db_path+db_name)
        cur = con.cursor()
        af_list = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
        sel_string = ", ".join([af_list[ix+1] for ix in range(len(af_list)-1)])
        audio_features_mins = np.array([config_vars['audio_features'][af]['min'] for af in af_list[1:]])
        audio_features_maxs = np.array([config_vars['audio_features'][af]['max'] for af in af_list[1:]])
        for record in cur.execute("SELECT %s, genres FROM track_attributes WHERE track_id='%s'" % (sel_string, uri)):
            af_vec = np.array([[record[ix] for ix in range(len(af_list)-1)]])
            af_vec = (af_vec-audio_features_mins)/(audio_features_maxs-audio_features_mins)
            genres = record[-1]
            genre_vec = np.mean(np.array([genre_model.wv[genre] for genre in genres.split(" ") if genre in genre_model.wv.index_to_key]), axis=0)
            if np.any(np.isnan(genre_vec)):
                genre_vec = np.zeros(5)
            genre_vec = 0.5+(np.array([genre_vec])/20.0)
        match_vec = np.array([[0,0]])
        comb_vec = np.concatenate([word_vec, genre_vec, af_vec, match_vec], axis=1)
        pred_vec = model.predict(comb_vec)[0]
        word_vec = 0.5+(wvmodel.wv[word]/20.0)
        return pred_vec

def get_descriptors(model, db_path, db_name, wordvectors_name, uri):
    wvmodel = Word2Vec.load(db_path+wordvectors_name)
    con = sqlite3.connect(db_path+db_name)
    cur = con.cursor()
    af_list = ['id','danceability','energy','key','loudness','mode','speechiness','acousticness','instrumentalness','liveness','valence','tempo']
    sel_string = ", ".join([af_list[ix+1] for ix in range(len(af_list)-1)])
    audio_features_mins = np.array([config_vars['audio_features'][af]['min'] for af in af_list[1:]])
    audio_features_maxs = np.array([config_vars['audio_features'][af]['max'] for af in af_list[1:]])
    for record in cur.execute("SELECT %s FROM track_attributes WHERE track_id='%s'" % (sel_string, uri)):
        af_vec = np.array([[record[ix] for ix in range(len(af_list)-1)]])
        af_vec = (af_vec-audio_features_mins)/(audio_features_maxs-audio_features_mins)
    pred_vec = model.predict(af_vec)[0]
    return [sim for sim in wvmodel.wv.similar_by_vector(pred_vec, topn=100) if not "0" in sim[0]]
            

def load_song_attributes_from_spotify(db_path, db_name, n_tracks=1000):
    con = sqlite3.connect(db_path+db_name)
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
            `genres` varchar(100) DEFAULT NULL,
            `popularity` int(10) DEFAULT NULL,
            FOREIGN KEY (`track_id`) REFERENCES `track` (`id`)
        );
        """)
    con.commit()
    cur.execute("""SELECT A.track_id,
                          B.artist_id
                     FROM track_playlist1 A
                     JOIN track_artist1 B
                       ON A.track_id = B.track_id
                    GROUP BY A.track_id, B.artist_id
                    ORDER BY COUNT(*) DESC LIMIT %i""" % n_tracks)
                    
    done = False
    results = cur.fetchall()
    n_ids = 50
    cur_ix = 0
    while not done:
        if len(results)-cur_ix <= n_ids:
            done = True
            n_ids = len(results)-(cur_ix+n_ids)
        result = results[cur_ix:cur_ix+n_ids]
        song_uris = [res[0] for res in result]
        art_uris = [res[1] for res in result]
        cur_ix += n_ids
        try:
            audio_features = make_spotify_request('audio-features',{'ids':",".join(song_uris)})['audio_features']
            artist_features = make_spotify_request('artists',{'ids':",".join(art_uris)})['artists']
            feature_strings = []
            for afix in range(len(audio_features)):
                if audio_features[afix] is not None:
                    features = [str(audio_features[afix][fname]) for fname in af_list]
                    if artist_features[afix] is not None:
                        features = features+["-".join(artist_features[afix]["genres"]).replace(" ","_").replace("-"," "), str(artist_features[afix]["popularity"])]
                    else:
                        features = features+["",""]
                    feature_string = "('"+"','".join(features)+"')"
                    feature_strings.append(feature_string)
            feature_string = ", ".join(feature_strings)
            insert_string = "INSERT INTO track_attributes VALUES "+feature_string+";"
            cur.execute(insert_string)
            con.commit()
        except:
            pass
