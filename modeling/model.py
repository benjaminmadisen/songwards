from gensim.models import Word2Vec
import os
import re
import pickle

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
