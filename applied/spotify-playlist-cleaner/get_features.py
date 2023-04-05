import api
import time
for playlist in api.get_playlists():
    if playlist.id in ["7BTXgyKj7H6Qd3qjcvSnpG", "2Cjb7gZLeDRQdSAhncWO8u"]:
        continue
    offset = 0
    while True:
        songs = api.get_playlist_songs(playlist.id, offset=offset)
        delta = 0
        for song in songs:
            print(song.id)
            api.get_song_feature(
                id=song.id
            )
            delta += 1
        offset += 10
        if delta == 0:
            break
