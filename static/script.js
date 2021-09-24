

Vue.component('display-song-item', {
    props: ['song'],
    template: `
      <div class="display-song-item">
        <button v-on:click="app.removeSelectedSong(song)">
          Remove
        </button>
        <img v-bind:src="song.image_url">{{ song.name }} - {{ song.artist }}: {{song.score}}
      </div>
    `
  })
  Vue.component('search-song-item', {
    props: ['song'],
    template: `
      <div class="search-song-item">
      <img v-bind:src="song.image_url">{{ song.name }} - {{ song.artist }}
        <button v-on:click="app.addSearchedSong(song)">
          Add
        </button>
      </div>
    `
  })
var app = new Vue({
    el: '#app',
    delimiters: ["[[", "]]"],
    data: {
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
          xhr.open('GET', '/score_songs?text='+this.wordText+'&uris='+this.getUris());
          xhr.responseType = 'json';
          xhr.send();
          this.currentWord = this.wordText;
          this.wordText = "";

          var app = this;
          xhr.onload = function() {
            let responseObj = xhr.response;
            for (uri in responseObj.scores){
              app.selectedSongs.filter(function(s) { return s.uri == uri})[0].score = responseObj.scores[uri];
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
