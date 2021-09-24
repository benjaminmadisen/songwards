Vue.component('song-item', {
    props: ['song'],
    template: "<li>no {{ song.text }}</li><button v-on:click='removeUri({{ song.uri }})'>Remove</button>"
  })
var app = new Vue({
    el: '#app',
    delimiters: ["[[", "]]"],
    data: {
      uri_in: "",
      selectedSongs: [
          
      ]
    },
    created: function () {
        var element = document.getElementById('uris');
        var text = element.innerText;
        var uris = text.split(",");
        uris.forEach(this.addNewSong);
    },
    methods: {
        addUriIn: function () {
            if (this.uri_in != ""){
            this.selectedSongs.push({
                uri: this.uri_in,
                text: this.uri_in
            });
            var formData = new FormData();
            formData.append("uri", this.uri_in);
            var req = new XMLHttpRequest();
            req.open("POST", "/add_uri");
            req.send(formData);
            this.uri_in = "";
        }
        },
        removeUri: function (uri) {
            this.selectedSongs = this.selectedSongs.filter(function(s) { return s.uri !== uri })
        },
        addNewSong: function (value, index, array) {
          if (value != ""){
          this.selectedSongs.push({
            uri: value,
            text: value
          });
          }
        }
      }
  })
