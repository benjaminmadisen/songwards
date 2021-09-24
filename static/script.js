

Vue.component('display-song-item', {
    props: ['song'],
    template: `
      <div class="display-song-item">
        {{ song.text }}
        <button v-on:click="app.removeSelectedSong(song)">
          Remove
        </button>
      </div>
    `
  })
  Vue.component('search-song-item', {
    props: ['song'],
    template: `
      <div class="search-song-item">
        {{ song.text }}
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
        addSearchedSong: function (song) {
          this.selectedSongs.push({
            uri: song.uri,
            text: song.text
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
