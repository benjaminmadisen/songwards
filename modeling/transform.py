import sqlite3
from main import make_spotify_request
from gensim.utils import simple_preprocess
import os

class SpotifyTrackFeatureGenerator:
    """ A class to transform the spotify DB to add track feature info.

    """
    def __init__(self, db_path:str, db_name:str, audio_feature_list:list, artist_feature_list:list):
        """ Creates a SpotifyTrackFeatureGenerator instance.

        Args:
            db_path (str): path to the local directory where the db should live.
            db_name (str): file name for the local db.
            audio_feature_list (list): list of audio features to place in track_attributes table
            artist_feature_list (list): list of artist features to place in track_attributes table

        """
        self.db_path = db_path
        self.db_name = db_name
        self.audio_feature_list = audio_feature_list
        self.artist_feature_list = artist_feature_list
    
    def create_track_feature_table(self, replace:bool=True) -> None:
        """ Creates the track feature table.

        Args:
            replace (bool): replace the table if it exists?

        """
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        if replace:
            cur.execute("DROP TABLE IF EXISTS `track_attributes`;")
        
        create_table_statement = "CREATE TABLE `track_attributes` ("
        create_table_statement += "`track_id` varchar(100) DEFAULT NULL,"
        for audio_feature in self.audio_feature_list + self.artist_feature_list:
            create_table_statement += "`%s` %s DEFAULT NULL," % (audio_feature, self.get_sqlite_type(audio_feature))
        create_table_statement += "FOREIGN KEY (`track_id`) REFERENCES `track` (`id`) );"
        cur.execute(create_table_statement)
        con.commit()
    
    def get_sqlite_type(self, feature:str) -> str:
        """ Returns sqlite dtype for feature.

        TODO: load from config, rather than hardcode

        Args:
            feature (str): feature to search for.
        """
        type_dict = {"danceability": "double",
                     "energy": "double",
                     "key": "int(4)",
                     "loudness": "double",
                     "mode": "int(4)",
                     "speechiness": "double",
                     "acousticness": "double",
                     "instrumentalness": "double",
                     "liveness": "double",
                     "valence": "double",
                     "tempo": "double",
                     "duration_ms": "int(10)",
                     "time_signature": "int(4)",
                     "genres": "varchar(100)",
                     "popularity": "int(10)"}
        return type_dict[feature]
            
    
    def create_popular_track_table(self, n_tracks:int=10000, replace:bool=True) -> None:
        """ Creates a table with the n_tracks most frequently included in playlists

        Args:
            n_tracks (int): number of tracks to include
            replace (bool): replace the table if it exists?

        """
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        if replace:
            cur.execute("DROP TABLE IF EXISTS `popular_tracks`;")
        cur.execute("""
                   CREATE TABLE `popular_tracks` AS
                   SELECT A.track_id,
                          B.artist_id
                     FROM track_playlist1 A
                     JOIN track_artist1 B
                       ON A.track_id = B.track_id
                    GROUP BY A.track_id, B.artist_id
                    ORDER BY COUNT(*) DESC LIMIT %i""" % n_tracks)
        con.commit()

    def lookup_track_attributes(self, lookup_table:str) -> None:
        """ Looks up track attributes of tracks in lookup_table

        Args:
            lookup_table (str): table name to lookup. Must have track_id and artist_id columns

        """
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        cur.execute("""
                   SELECT A.track_id,
                          A.artist_id
                     FROM %s A
                     JOIN track_attributes B
                       ON A.track_id = B.track_id
                    WHERE B.track_id IS NULL""" % lookup_table)
        done = False
        while not done:
            # Max artists in spotify request
            uris = cur.fetchmany(50)
            song_uris = [uri[0] for uri in uris]
            art_uris = [uri[1] for uri in uris]
            # if spotify errors, keep going for another 50
            try:
                audio_features = make_spotify_request('audio-features',{'ids':",".join(song_uris)})['audio_features']
                artist_features = make_spotify_request('artists',{'ids':",".join(art_uris)})['artists']
                insert_string = self.get_insert_string(audio_features, artist_features)
                cur.execute(insert_string)
                con.commit()
            except:
                pass

    def get_insert_string(self, audio_features:dict, artist_features:dict) -> str:
        """ Get string to use in SQLite INSERT clause using feature dictionaries.

        Args:
            audio_features: result of spotify audio-features request
            artist_features: result of spotify artists request

        """
        feature_strings = []
        for ix in range(len(audio_features)):
            if audio_features[ix] is not None:
                features = [str(audio_features[ix][feature_name]) for feature_name in self.audio_feature_list]
                if artist_features[ix] is not None:
                    for feaure_name in self.artist_feature_list:
                        feature = str(artist_features[ix][feaure_name])
                        if feature == "genres":
                            feature = "-".join(feature.replace(" ","_")).replace("-"," ")
                        features = features+[feature]
                else:
                    features = features+["" for feature_name in self.artist_feature_list]
                feature_string = "('"+"','".join(features)+"')"
                feature_strings.append(feature_string)
        return "INSERT INTO track_attributes VALUES "+", ".join(feature_strings)+";"


class SpotifyCorpusTextGenerator:
    """ A class to generate files as corpus inputs to Word2Vec.

    """
    def __init__(self, db_path:str, db_name:str):
        """ Creates a SpotifyCorpusTextGenerator instance.

        Args:
            db_path (str): path to the local directory where the db should live.
            db_name (str): file name for the local db.

        """
        self.db_path = db_path
        self.db_name = db_name
    
    def generate_random_playlist_sample(self, n_playlists:int=100000):
        """ Add a table of n_playlists random playlists and their track pairs

        Args:
            n_playlists (int): number of playlists to sample

        """
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        cur.execute("DROP TABLE IF EXISTS temp_playlist")
        cur.execute("""CREATE TABLE temp_playlist
                           AS SELECT A.id, A.name
                         FROM playlist A 
                        ORDER BY RANDOM() LIMIT %s""" % n_playlists)
        con.commit()
        cur.execute("DROP TABLE IF EXISTS temp_track_playlist")
        cur.execute("""CREATE TABLE temp_track_playlist
                           AS SELECT A.id, A.name, C.*
                         FROM temp_playlist A
                         JOIN track_playlist1 B ON A.id = B.playlist_id
                         JOIN track_attributes C ON B.track_id = C.track_id""")
        con.commit()
    
    def generate_word_vector_text(self, text_path:str, replace:bool=True):
        """ Generates a file at text_path for use in building word vectors.

        Args:
            text_path (str): location for text output.
            replace (bool): should the file be replaced if it exists?

        """
        if os.path.exists(self.db_path+text_path):
            if replace:
                os.remove(self.db_path+text_path)
            else:
                raise FileExistsError()
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        with open(self.db_path+text_path, "w") as tf:
            for row in cur.execute("SELECT track_id, name FROM temp_track_playlist"):
                tf.write(" ".join([row[0]] + simple_preprocess(row[1], deacc=True))+" \n")
    
    def generate_genre_vector_text(self, text_path:str, replace:bool=True):
        """ Generates a file at text_path for use in building genre vectors.

        Args:
            text_path (str): location for text output.
            replace (bool): should the file be replaced if it exists?

        """
        if os.path.exists(self.db_path+text_path):
            if replace:
                os.remove(self.db_path+text_path)
            else:
                raise FileExistsError()
        con = sqlite3.connect(self.db_path+self.db_name)
        cur = con.cursor()
        with open(self.db_path+text_path, "w") as tf:
            cur_genres = []
            cur_track_id = -1
            for row in cur.execute("SELECT id, genres FROM temp_track_playlist"):
                if row[0] != cur_track_id:
                    tf.write(" ".join(cur_genres)+"\n")
                    cur_genres = []
                for genre in row[1].split(" "):
                    cur_genres.append(genre)


            
