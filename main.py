from flask import Flask, render_template, session, redirect, url_for, request
from time import time
import os

app = Flask(__name__)
app.secret_key = os.urandom(50)

@app.route('/')
def root():
    uris = ""
    if 'songbird_recent' in session:
        if time()-int(session['songbird_recent']) < 3600:
            if 'songbird_uris' in session:
                uris = session['songbird_uris']
    session['songbird_recent'] = time()
    return render_template('index.html', uris=uris)

@app.route('/add_uri', methods=['POST'])
def add_uri():
    if 'songbird_uris' not in session:
        session['songbird_uris'] = request.form['uri']+","
    elif request.form['uri'] not in session['songbird_uris']:
        session['songbird_uris'] += request.form['uri']+","
    return redirect(url_for('root'))
    

@app.route('/remove_uri', methods=['POST'])
def remove_uri():
    if 'songbird_uris' in session:
        if request.form['uri'] in session['songbird_uris']:
            session['songbird_uris'] = session['songbird_uris'].replace(request.form['uri']+",","")
    return redirect(url_for('root'))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8080, debug=True)