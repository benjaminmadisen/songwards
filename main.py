from flask import Flask, render_template, session, redirect, url_for, request
from google.cloud import secretmanager
from time import time
import os
import requests
import numpy as np

client = secretmanager.SecretManagerServiceClient()

app = Flask(__name__)
app.secret_key = os.urandom(50)

global_access_token = None

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
            'score': -1}

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

def score_uris(search_text, uris):
    rand_vals = np.random.random(len(uris))
    out = {}
    for uri_ix in range(len(uris)):
        out[uris[uri_ix]] = rand_vals[uri_ix]
    return out

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

@app.route('/score_songs')
def score_songs():
    search_text = request.args.get('text', None)
    uris = request.args.get('uris', None).split(",")
    if search_text is not None:
        if len(uris) > 0:
            return {'scores': score_uris(search_text, uris)}
    return {'scores':{}}

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

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)