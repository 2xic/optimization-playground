import api

class Dataset:
    def __init__(self) -> None:
        self.ids_to_name = {}
        self.ids_features = {}
        self.ids_songs = {}
        self.id_class = {}
        self.class_id = {}

    def load(self, exclude=[], include=None, minimum=5):
        count = 0
        for playlist in api.get_playlists():
            if include is not None and playlist.id not in include:
                continue
            if playlist.id in exclude:
                continue
            self.load_playlist(playlist.id)
            count += 1
            if count > minimum:
                break
        return self
    
    def load_playlist(self, id):
        if id not in self.ids_features:
            for playlist in api.get_playlists():
                if playlist.id == id:
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
            print(train, testing, len(songs))
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
        song_name = []
        for song, audio_features in zip(self.ids_songs[playlist_id], self.ids_features[playlist_id]):
            song_name.append(song.name)
            x.append([
                getattr(audio_features, feature)
                for feature in features
            ])
        return x, song_name

    def _get_songs(self, playlist_id):
        offset = 0
        features = []
        all_songs = []
        while True:
            songs = api.get_playlist_songs(playlist_id, offset=offset)
            delta = 0
            for song in songs:
                i = list(api.get_song_feature(id=song.id))[0]
                features.append(i)
                all_songs.append(song)
                delta += 1
                offset += 1
            offset += len(songs)
            if delta == 0:
                break
        return all_songs, features
    