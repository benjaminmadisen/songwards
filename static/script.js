
Vue.component('display-song-item', {
    props: ['song'],
    template: `
      <div class="display-song-item">
        <button class="add_remove" v-on:click="app.removeSelectedSong(song)">
          -
        </button>
        <div class="song-info">
          <img v-bind:src="song.image_url">{{ song.name }} - {{ song.artist }}
        </div>
        <div class="song-score">
          {{song.score}}
        </div>
      </div>
    `
  })
  Vue.component('search-song-item', {
    props: ['song'],
    template: `
      <div class="search-song-item">
        <div class="song-info">
          <img v-bind:src="song.image_url">{{ song.name }} - {{ song.artist }}
        </div>
        <button class="add_remove" v-on:click="app.addSearchedSong(song)">
          +
        </button>
      </div>
    `
  })
var app = new Vue({
    el: '#app',
    delimiters: ["[[", "]]"],
    data: {
      model: false,
      wordVector: [0,0,0,0,0,0,0,0,0,0],
      searchText: "",
      wordText: "",
      currentWord: "",
      selectedSongs: [],
      searchedSongs: []
    },
    methods: {
        searchSongs: function () {
          let xhr = new XMLHttpRequest();
          xhr.open('GET', '/search_songs?text='+this.searchText);
          xhr.responseType = 'json';
          xhr.send();
          this.searchText = "";

          var app = this;
          xhr.onload = function() {
            let responseObj = xhr.response;
            app.searchedSongs = responseObj.songs;
          };
        },
        searchWord: function () {
          let xhr = new XMLHttpRequest();
          xhr.open('GET', '/get_text_vector?text='+this.wordText);
          xhr.responseType = 'json';
          xhr.send();
          this.currentWord = this.wordText;
          this.wordText = "";

          var app = this;
          xhr.onload = function() {
            let responseObj = xhr.response;
            app.wordVector = responseObj.vector;
            for (song_id in app.selectedSongs){
              let song = app.selectedSongs[song_id];
              inp_array = app.wordVector.concat(song.vector[0]);
              if (song.name.toLowerCase().includes(app.currentWord.toLowerCase())){
                inp_array[inp_array.length-2] = 1;
              }
              if (song.artist.toLowerCase().includes(app.currentWord.toLowerCase())){
                inp_array[inp_array.length-1] = 1;
              }
              inpvec = tf.tensor([inp_array]);
              app.model.predict(inpvec).array().then(array => song.score = array[0][0]);
            }
          };
        },
        getUris: function () {
          let out = "";
          for (song in this.selectedSongs){
            out = out+this.selectedSongs[song].uri+",";
          }
          out = out.substring(0, out.length - 1);
          return out;
        },
        addSearchedSong: function (song) {
          this.selectedSongs.push({
            uri: song.uri,
            name: song.name,
            artist: song.artist,
            image_url: song.image_url,
            vector: song.vector,
            score: -1
          });
          var formData = new FormData();
          formData.append("uri", song.uri);
          var req = new XMLHttpRequest();
          req.open("POST", "/add_uri");
          req.send(formData);
          this.searchedSongs = this.searchedSongs.filter(function(s) { return s.uri !== song.uri })
        },
        removeSelectedSong: function (song) {
          var formData = new FormData();
          formData.append("uri", song.uri);
          var req = new XMLHttpRequest();
          req.open("POST", "/remove_uri");
          req.send(formData);
          this.selectedSongs = this.selectedSongs.filter(function(s) { return s.uri !== song.uri })
        }
      },
      created: function () {
        let temp_app = this;
        tf.loadLayersModel('model_info/model.json').then(
          function(value) {temp_app.model = value;}
        );
        let xhr = new XMLHttpRequest();
        xhr.open('GET', '/get_songs');
        xhr.responseType = 'json';
        xhr.send();

        let selectedSongs = this.selectedSongs;
        xhr.onload = function() {
          let responseObj = xhr.response;
          for (const song of responseObj.songs){
            selectedSongs.push(song);
          }
        };
        
      }
  })
