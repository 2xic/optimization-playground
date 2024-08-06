# playlists
# playlist/id/
# -> For each song in playlist get the feature
import requests
from collections import namedtuple
import json
import hashlib
import os
from dotenv import load_dotenv
import time
import urllib3
import time, os, stat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

load_dotenv()

assert os.environ["cookie"], "No cookie?"

class Api:
    def __init__(self, is_cache_only_mode) -> None:
        self.config = None
        with open("config.json", "r") as file:
            self.config = json.load(file)
        self.api_url = self.config["host"]
        self.one_hour = 60 * 60
        self.playlist_max_age = 12 * self.one_hour if not is_cache_only_mode else float('inf')
        self.playlist = namedtuple('playlist', ['id', 'name'])
        self.song = namedtuple('song', ['id', 'name', 'image'])
        self.feature = namedtuple('features', [
            'danceability', 
            'energy', 
            'key', 
            'loudness', 
            'mode', 
            'speechiness', 
            'acousticness', 
            'instrumentalness', 
            'liveness', 
            'valence', 
            'tempo'
        ])
        self.is_cache_only_mode = is_cache_only_mode

    def get_cache(self, url):
        hash = hashlib.sha256(url.encode()).hexdigest()[:8]
        dir_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            ".cache",
        )
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(
            dir_path,
            hash
        )
        return path

    def play_song(self, song_id):
        data = requests.post(f"{self.api_url}/play/{song_id}", verify=False, headers={
            "Cookie": "auth_token=" + os.getenv("cookie")
        })
        print(data)

    def move_song(self, song_id, from_playlist, to_playlist):
        data = requests.post(f"{self.api_url}/playlist/move/{from_playlist}/{to_playlist}/{song_id}", verify=False, headers={
            "Cookie": "auth_token=" + os.getenv("cookie")
        })
        print(data)

    def get_file_age_in_seconds(self, pathname):
        return time.time() - os.stat(pathname)[stat.ST_MTIME]

    def get_requests_cache(self, url, max_age=float('inf')):
        path = self.get_cache(url)
        if os.path.isfile(path):
            if self.get_file_age_in_seconds(path) < max_age:
                with open(path, "r") as file:
                    content = file.read()
                    if len(content):
                        results = json.loads(content)
                        for keys in results:
                            if results[keys] != None:
                                return results
                        #if results.get("items", None) != None or results.get("audio_features", None) != None:
                        #    return results
        # We don't have the file, we can't fetch it in cache only mode.
        if self.is_cache_only_mode:
            return None
        print(f"Requesting: {url}")
        data = requests.get(url, verify=False, headers={
            "Cookie": "auth_token=" + os.getenv("cookie")
        })
        print(f"status code ... {data.status_code}")
        assert data.status_code == 200, f"Internal error, maybe rate liming ? Status code: {data.status_code}, Url: {url}" 
        data = data.json()
        if data is None or data.get("items", True) == None:
            print("Got none from the api, have you added the cookie ?")
            exit(0)
        time.sleep(3)
        with open(path, "w") as file:
            file.write(json.dumps(data))
        return data

    def get_playlist(self, id):
        playlist = self.get_requests_cache(f"{self.api_url}/playlist/{id}")
        # TODO: Add importing of name
        return self.playlist(id=id, name="unknown playlist name?")

    def get_playlists(self):
        for i in self.get_requests_cache(f"{self.api_url}/playlists")["items"]:
            if i["owner"]["display_name"].lower() != self.config["user_display_name"].lower():
                continue
            yield self.playlist(id=i["id"], name=i["name"])

    def get_playlist_songs(self, id, offset=0):
        print(f"{self.api_url}/playlist/{id}?offset={offset}")
        for i in self.get_requests_cache(f"{self.api_url}/playlist/{id}?offset={offset}", max_age=self.playlist_max_age)["items"]:
            images = i["track"]["album"]["images"]
            url = None
            if not images is None:
                url = (images[-1]["url"])
            yield self.song(id=i["track"]["id"], name=i["track"]["name"], image=url)

    def get_song_feature(self, id):
        print(f"{self.api_url}/song/{id}/feature")
        feature = self.get_requests_cache(f"{self.api_url}/song/{id}/feature")
        audio_features = feature["audio_features"]
        for i in audio_features:
            yield self.feature(
                danceability=i["danceability"],
                energy=i['energy'], 
                key=i['key'], 
                loudness=i['loudness'], 
                mode=i['mode'], 
                speechiness=i['speechiness'], 
                acousticness=i['acousticness'], 
                instrumentalness=i['instrumentalness'], 
                liveness=i['liveness'], 
                valence=i['valence'], 
                tempo=i['tempo']
            )

    def get_artists_top_track_playlist(self, id):
        data = self.get_requests_cache(f"{self.api_url}/artists/{id}/top_tracks")
        print(data)
        return data
