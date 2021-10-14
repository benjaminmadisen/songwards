from gensim.models import Word2Vec
import os
import re
import pickle
import numpy as np
import tensorflow as tf
import tensorflowjs as tfjs

def generate_tf_dataset_from_files(db_path):
    ix = 1
    db_path = str(db_path)[2:-1]
    while ix < 200:
        if os.path.exists("%sbin_targets_%i.npy" % (db_path, ix)):
            with open("%sbin_targets_%i.npy" % (db_path, ix), "rb") as f:
                targets = np.load(f)
            with open("%sbin_trains_%i.npy" % (db_path, ix), "rb") as f:
                af_vecs = np.load(f)
            for rix in range(targets.shape[0]-1):
                yield af_vecs[rix,:], [targets[rix]]
        ix += 1

class TextCorpus:
    """ A simple iterator used by gensim's Word2Vec class.

    """

    def __init__(self, db_path:str, text_path:str):
        """ Returns a TextCorpus at db_path+text_path

        Args:
            db_path (str): directory containing text
            text_path (str): file location of text file

        """
        self.db_path = db_path
        self.text_path = text_path

    def __iter__(self):
        with open(self.db_path+self.text_path, "r") as tf:
            for row in tf.readlines():
                sentence = row.replace("\n","").split(" ")
                yield sentence


class WordvectorModel:
    """ A wrapper for training and saving gensim Word2Vec models.

    """
    
    def __init__(self, db_path:str):
        """ Returns a WordvectorModel.

        Args:
            db_path (str): directory containing text file

        """
        self.db_path = db_path
        self.model = None
    
    def train_model(self, text_path:str, vector_size:int=5, min_count:int=5) -> None:
        """ Trains and potentially saves a Word2Vec model.

        Args:
            text_path (str): location of the text corpus.
            vector_size (int): size of output vector.
            min_count (int): minimum number of examples in corpus to be included in analysis.

        """
        sentences = TextCorpus(self.db_path,text_path)
        self.model = Word2Vec(sentences=sentences, vector_size=vector_size, min_count=min_count)
    
    def save_model_to_local(self, save_path:str, replace:bool=True) -> None:
        """ Saves a pre-trained model locally.

        Args:
            save_path (str): location to save model.
            replace (bool): should we replace an already saved model?

        """
        if self.model is None:
            raise RuntimeError("Model not yet trained.")
        if os.path.exists(self.db_path+save_path):
            if replace:
                os.remove(self.db_path+save_path)
            else:
                raise FileExistsError()
        self.model.save(self.db_path+save_path)
    
    def save_model_to_gcloud(self, gcloud_path:str, storage_bucket)-> None:
        """ Saves a pre-trained model to gcloud.

        Args:
            save_path (str): location to save model.
            storage_bucket: a google cloud storage bucket object

        """
        word_vecs = self.model.wv
        useful_words = [itk for itk in word_vecs.index_to_key if not re.search(r'\d', itk)]
        out_dict = {}
        for word in useful_words:
            out_dict[word] = word_vecs[word]
        blob = storage_bucket.blob(gcloud_path)
        blob.upload_from_string(pickle.dumps(out_dict))

class SimpleMatchModel:
    """ A simple model, estimating probability that a pair of (playlist name, track) is from the data, or random.

    """
    
    def __init__(self, db_path:str, data_path:str):
        """ Returns an instance of SimpleMatchModel.

        Args:
            db_path (str): directory containing output data
            data_path (str): dir name within db_path of training data.

        """
        self.db_path = db_path
        self.train_data_path = data_path+"train/"
        self.val_data_path = data_path+"val/"
        self.model = None
    
    def generate_tf_dataset(self, input_vector_length:int):
        """ Returns a tf Dataset based on input path info.

        Args:
            input_vector_length: length of input vector
        
        """
        return tf.data.Dataset.from_generator(
            generate_tf_dataset_from_files,
            args=[self.db_path+self.train_data_path],
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape((input_vector_length,)), tf.TensorShape((1,)))), \
            tf.data.Dataset.from_generator(
            generate_tf_dataset_from_files,
            args=[self.db_path+self.val_data_path],
            output_types=(tf.float32, tf.float32),
            output_shapes=(tf.TensorShape((input_vector_length,)), tf.TensorShape((1,))))

    def init_model(self, model_structure=None):
        """ Instantiates a simple keras sequential model.

        Args:
            model_structure: an uncompiled keras Sequential model, or None to use default.

        """
        if model_structure is None:
            self.model = tf.keras.Sequential([
                tf.keras.Input(shape=(28,)),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(64, activation='relu'),
                tf.keras.layers.Dense(2, activation='softmax')])
        else:
            self.model = model_structure
        self.model.compile(
            optimizer="adam",
            loss=tf.keras.losses.SparseCategoricalCrossentropy(), 
            metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
    
    def train_model(self, input_vector_length:int=28, epochs:int=5, callbacks=None):
        """ Trains the model for epochs epochs.

        Args:
            epochs: number of epochs to train for.
            input_vector_length: length of input vector
            callbacks: a tf.keras callbacks list to pass to model.fit

        """
        train_dataset, val_dataset = self.generate_tf_dataset(input_vector_length)
        train_dataset = train_dataset.batch(32).shuffle(buffer_size=1000).repeat()
        val_dataset = val_dataset.batch(32).shuffle(buffer_size=1000).repeat()
        if callbacks is None:
            self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=5000, validation_steps=1000)
        else:
            self.model.fit(train_dataset, epochs=epochs, validation_data=val_dataset, steps_per_epoch=5000, validation_steps=1000, callbacks=callbacks)
    
    def save_model_to_local(self, save_path:str, replace:bool=True) -> None:
        """ Saves a pre-trained model locally.

        Args:
            save_path (str): location to save model.
            replace (bool): should we replace an already saved model?

        """
        if os.path.isdir(self.db_path+save_path):
            if replace:
                dir_files = os.listdir(self.db_path+save_path)
                for file in dir_files:
                    os.remove(self.db_path+save_path+file)
                os.rmdir(self.db_path+save_path)
            else:
                raise FileExistsError()
        tfjs.converters.save_keras_model(self.model, self.db_path+save_path)

    def save_model_to_gcloud(self, save_path: str, gcloud_paths:dict, storage_bucket)-> None:
        """ Saves a pre-trained model to gcloud.

        Args:
            save_path (str): location of saved model.
            gcloud_paths (dict): mapping from file names to gcloud file names.
            storage_bucket: a google cloud storage bucket object.

        """
        model_outputs = os.listdir(self.db_path+save_path)
        for model_output in model_outputs:
            if model_output in gcloud_paths:
                blob = storage_bucket.blob(gcloud_paths[model_output])
                blob.upload_from_filename(self.db_path+save_path+model_output)
            else:
                print(model_output)
            os.remove(self.db_path+save_path+model_output)
        os.rmdir(self.db_path+save_path)