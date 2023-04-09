import api
import time
import json

if __name__ == "__main__":
    with open("config.json", "r") as file:
        json_data = json.loads(file.read())
        exclude = json_data["exclude"]
        for playlist_id in json_data["include"]:
            offset = 0
            while True:
                songs = api.get_playlist_songs(playlist_id, offset=offset)
                delta = 0
                for song in songs:
                    api.get_song_feature(
                        id=song.id
                    )
                    delta += 1
                    offset += 1
                if delta == 0:
                    break
