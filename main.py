from flask import Flask, render_template, session, redirect, url_for, request
from google.cloud import secretmanager, storage
from time import time
import os
import requests
import numpy as np
import yaml
import pickle
import json

client = secretmanager.SecretManagerServiceClient()

app = Flask(__name__)
app.secret_key = os.urandom(50)

global_access_token = None
global_interpreter = None
global_wordvecs = None
global_genrevecs = None
global_cache = {}
with open(".songwards_config", 'r') as config_file:
    config_vars = yaml.load(config_file, Loader=yaml.CLoader)
audio_features_list = list(config_vars['audio_features'].keys())
audio_features_mins = np.array([0,0,0,0,0]+[config_vars['audio_features'][af]['min'] for af in audio_features_list]+[0,0])
audio_features_maxs = np.array([1,1,1,1,1]+[config_vars['audio_features'][af]['max'] for af in audio_features_list]+[1,1])
valid_tfjs_paths = config_vars['valid_tfjs_paths']
storage_client = storage.Client()
bucket = storage_client.bucket(config_vars['bucket_path'])


def get_file_from_blob(file_name):
    global global_cache
    valid_path = valid_tfjs_paths[file_name]
    if valid_path not in global_cache:
        blob = bucket.blob(valid_path)
        global_cache[valid_path] = blob.download_as_bytes()
    return global_cache[valid_path]

def get_wordvecs():
    global global_wordvecs
    if global_wordvecs is None:
        blob = bucket.blob(config_vars['wordvecs_path'])
        global_wordvecs = pickle.loads(blob.download_as_bytes())
        
    return global_wordvecs

def get_genrevecs():
    global global_genrevecs
    if global_genrevecs is None:
        blob = bucket.blob(config_vars['genrevecs_path'])
        global_genrevecs = pickle.loads(blob.download_as_bytes())
        
    return global_genrevecs

def get_gcloud_secret(secret_id):
    name = "projects/songwards/secrets/%s/versions/latest" % secret_id
    response = client.access_secret_version(request={"name": name})
    return response.payload.data.decode("UTF-8")

def get_spotify_keys():
    return get_gcloud_secret('SPOTIFY_CLIENT_ID'), get_gcloud_secret('SPOTIFY_CLIENT_SECRET')

def refresh_access_token():
    spotify_keys = get_spotify_keys()
    payload = {'grant_type':'client_credentials',
               'client_id':spotify_keys[0],
               'client_secret':spotify_keys[1]}
    r = requests.post("https://accounts.spotify.com/api/token/", data=payload).json()
    return r['access_token']

def get_access_token():
    global global_access_token
    if global_access_token is None:
        access_token = refresh_access_token()
    return access_token

def get_song_object_from_track_item(item):
    return {'uri': item['id'],
            'name': item['name'],
            'artist': item['artists'][0]['name'],
            'image_url': [img['url'] for img in item['album']['images'] if img['height'] < 100][0],
            'vector': get_track_input(item['id']).astype(float).tolist(),
            'score': -1,
            'score_color': 'none'}

def make_spotify_request(endpoint, payload):
    access_token = get_access_token()
    headers = {'Authorization': 'Bearer '+access_token}
    r = requests.get("https://api.spotify.com/v1/%s?%s" % (endpoint, "&".join(["%s=%s" % (p_key, payload[p_key]) for p_key in list(payload.keys())])), headers=headers)
    try:
        r = r.json()
    except:
        return {}
    if 'error' in r:
        if 'message' in r['error']:
            if r['error']['message'] == 'The access token expired':
                global global_access_token
                global_access_token = None
                return make_spotify_request(endpoint, payload)
    return r

def lookup_spotify(uris):
    payload = {'ids':",".join(uris)}
    r = make_spotify_request('tracks', payload)
    if 'tracks' in r:
        return [get_song_object_from_track_item(item) for item in r['tracks']]
    return []

def search_spotify(search_text):
    payload = {'q':search_text,
               'type':'track',
               'limit':'3'}
    r = make_spotify_request('search', payload)
    if 'tracks' in r:
        if 'items' in r['tracks']:
            return [get_song_object_from_track_item(item) for item in r['tracks']['items']]
    return []

def get_track_input_from_spotify(uri):
    global global_cache
    genrevecs = get_genrevecs()
    audio_features = make_spotify_request('audio-features',{'ids':uri})['audio_features']
    track_info = make_spotify_request('tracks',{'ids':uri})['tracks']
    artist_features = make_spotify_request('artists',{'ids':track_info[0]['artists'][0]['id']})['artists']
    genres = artist_features[0]["genres"]
    genre_vec = [0,0,0,0,0]
    if len(genres) > 0:
        any_genre = False
        for genre in genres:
            if genre in genrevecs:
                any_genre = True
        if any_genre:
            genre_vec = (0.5+(np.mean(np.array([genrevecs[genre] for genre in genres if genre in genrevecs]), axis=0)/20.0)).tolist()

    af_vec = np.array([genre_vec+[track[keyname] for keyname in audio_features_list] +[0,0] for track in audio_features])
    af_vec = (af_vec-audio_features_mins)/(audio_features_maxs-audio_features_mins)
    return af_vec

def get_track_input(uri):
    global global_cache
    if uri in global_cache:
        return global_cache[uri]
    else:
        return get_track_input_from_spotify(uri)

def get_text_input(search_text):
    wv = get_wordvecs()
    if search_text in wv:
        return (.5+wv[search_text].astype(float)/20.0).tolist()
    else:
        return None

@app.route('/')
def root():
    uris = ""
    if 'songwards_recent' in session:
        if time()-int(session['songwards_recent']) > 3600:
            if 'songwards_uris' in session:
                session.pop('songwards_uris')
    session['songwards_recent'] = time()
    return render_template('index.html', uris=uris)

@app.route('/get_songs')
def session_songs():
    songs = []
    if 'songwards_uris' in session:
        if len(session['songwards_uris']) > 0:
            uris = session['songwards_uris'].split(",")[:-1]
            songs = lookup_spotify(uris)
    return {'songs':songs}

@app.route('/search_songs')
def search_songs():
    search_text = request.args.get('text', None)
    if search_text is not None:
        tracks = search_spotify(search_text)
        return {'songs':tracks}
    return {'songs':[]}

@app.route('/add_uri', methods=['POST'])
def add_uri():
    if 'songwards_uris' not in session:
        session['songwards_uris'] = request.form['uri']+","
    elif request.form['uri'] not in session['songwards_uris']:
        session['songwards_uris'] += request.form['uri']+","
    return redirect(url_for('root'))

@app.route('/remove_uri', methods=['POST'])
def remove_uri():
    if 'songwards_uris' in session:
        if request.form['uri'] in session['songwards_uris']:
            session['songwards_uris'] = session['songwards_uris'].replace(request.form['uri']+",","")
    return redirect(url_for('root'))

@app.route('/model_info/<file_name>')
def model_info(file_name):
    if file_name in list(valid_tfjs_paths.keys()):
        blob_file = get_file_from_blob(file_name)
        if '.json' in file_name:
            return json.loads(blob_file)
        return blob_file
    return None

@app.route('/get_text_vector')
def get_text_vector():
    search_text = request.args.get('text', None)
    if search_text is not None:
        vector = get_text_input(search_text)
        if vector is not None:
            return {'vector': get_text_input(search_text)}
    return {'vector':False}


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)