from api import Api

class Dataset:
    def __init__(self) -> None:
        self.ids_to_name = {}
        self.ids_features = {}
        self.ids_songs = {}
        self.id_class = {}
        self.class_id = {}
        self.api = Api(is_cache_only_mode=True)

    def load(self, exclude=[], include=None, minimum=10):
        count = 0
        playlists = self.api.get_playlists()
        counter = 0
        for playlist in playlists:
            if include is not None and playlist.id not in include:
                continue
            if playlist.id in exclude:
                continue
            self.load_playlist(playlist.id)
            count += 1
            if count > minimum:
                break
            counter += 1
        assert counter > 0, "Found no playlists"
        return self
    
    def load_playlist(self, id):
        print(f"Loading playlist {id} ...")
        if id not in self.ids_features:
            playlist = self.api.get_playlist(id)
            assert playlist.id == id
            self.ids_to_name[playlist.id] = playlist.name
            songs, features = self._get_songs(playlist.id)
            self.ids_features[playlist.id] = features
            self.ids_songs[playlist.id] = songs

            index = len(self.id_class)
            self.id_class[playlist.id] = index
            self.class_id[index] = playlist.id
        return self
    
    def get_x_y(self, features=[], split=0.8, adjust_n_samples=False):
        x = []
        y = []
        x_test = []
        y_test = []

        n_sample_size = float('inf')
        
        song_playlist_ids = [

        ]
        for playlist_id, songs in self.ids_features.items():
            song_playlist_ids.append(
                [
                    songs,
                    playlist_id
                ]
            )
            n_sample_size = min(n_sample_size, len(songs))

        for songs, playlist_id in song_playlist_ids:
            delta_songs_split_train = int(len(songs) * split)
            delta_songs_split_testing = int(len(songs) * (1 - split))
            
            train = min(delta_songs_split_train, int(n_sample_size * split)) if adjust_n_samples else delta_songs_split_train
            testing = min(delta_songs_split_testing, int(n_sample_size * (1 - split))) if adjust_n_samples else delta_songs_split_testing
            #print(train, testing, len(songs))
            for index, i in enumerate(songs):
                if index < train:
                    x.append([
                        getattr(i, feature)
                        for feature in features
                    ])
                    y.append(self.id_class[playlist_id])
                elif (index) < (train + testing):
                    x_test.append([
                        getattr(i, feature)
                        for feature in features
                    ])
                    y_test.append(
                        self.id_class[playlist_id]
                    )
        return x, y, x_test, y_test
    
    def get_song_prediction(self, playlist_id, features):
        x = []
        songs = []
        for song, audio_features in zip(self.ids_songs[playlist_id], self.ids_features[playlist_id]):
            songs.append(song)
            x.append(self.get_song_features(
                audio_features=audio_features,
                features=features,
            ))
        return x, songs

    def get_song_features(self, audio_features, features):
        return [
            getattr(audio_features, feature)
            for feature in features
        ]

    def _get_songs(self, playlist_id):
        offset = 0
        features = []
        all_songs = []
        while True:
            songs = self.api.get_playlist_songs(playlist_id, offset=offset)
            delta = 0
            for song in songs:
                print(song.name)
                song_features = self.api.get_song_feature(id=song.id)
                if song_features is None:
                    continue
                song_features = list(song_features)
                if len(song_features) == 0:
                    continue
                # since all items are part of an array
                features.append(list(song_features)[0])
                all_songs.append(song)
                delta += 1
            offset += delta
            if delta == 0:
                break
        return all_songs, features
    