
Vue.component('display-song-item', {
    props: ['song'],
    template: `
      <div class="display-song-item">
        <div class="song-info">
          <button class="add_remove" v-on:click="app.removeSelectedSong(song)">
            <span class="material-icons md-36">&#xe15c;</span>
          </button>
          <img v-bind:src="song.image_url">{{ song.name }} - {{ song.artist }}
        </div>
        <div v-show="song.score_color != 'none'" v-bind:style="{ color: song.score_color }" class="song-score">
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
          <span class="material-icons md-36">&#xe147;</span>
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
      errorMessage: "",
      songErrorMessage: "",
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
          xhr.open('GET', '/get_text_vector?text='+this.wordText.toLowerCase());
          xhr.responseType = 'json';
          xhr.send();
          this.currentWord = this.wordText;
          this.wordText = "";
          this.errorMessage = "";

          var app = this;
          xhr.onload = function() {
            let responseObj = xhr.response;
            app.wordVector = responseObj.vector;
            console.log(responseObj.vector);
            if (responseObj.vector == false){
              app.errorMessage = "sorry, we don't have data for this word";
              for (song_id in app.selectedSongs){
                let song = app.selectedSongs[song_id];
                song.score = -1;
                song.score_color = "none";
              }
            }
            else{
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
              app.model.predict(inpvec).array().then(array => {
                song.score = Math.round(100*array[0][1]);
                redscore = Math.round(228-2*song.score)
                greenscore = Math.round(28+2*song.score)
                song.score_color = "rgb("+redscore+","+greenscore+",128)";
              });
            }}
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
          if (this.selectedSongs.length < 10) {
            this.selectedSongs.push({
              uri: song.uri,
              name: song.name,
              artist: song.artist,
              image_url: song.image_url,
              vector: song.vector,
              score: -1,
              score_color: "none"
            });
            var formData = new FormData();
            formData.append("uri", song.uri);
            var req = new XMLHttpRequest();
            req.open("POST", "/add_uri");
            req.send(formData);
            this.searchedSongs = this.searchedSongs.filter(function(s) { return s.uri !== song.uri })
          }
          else {
            this.songErrorMessage = "maximum of ten active songs";
          }

        },
        removeSelectedSong: function (song) {
          var formData = new FormData();
          formData.append("uri", song.uri);
          var req = new XMLHttpRequest();
          req.open("POST", "/remove_uri");
          req.send(formData);
          this.selectedSongs = this.selectedSongs.filter(function(s) { return s.uri !== song.uri })
          this.songErrorMessage = "";
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
