from dotenv import load_dotenv
import hashlib
import time, os, stat
import json
import requests
from collections import namedtuple

load_dotenv()

class Api:
    def __init__(self) -> None:
        self.config = None
        with open("config.json", "r") as file:
            self.config = json.load(file)
        self.api_url = self.config["host"]
        # Simple interface
        self.song = namedtuple('song', ['id', 'name', 'artist'])
        self.playing = namedtuple('playing', ['progress', 'track'])

    def get_playlist(self, id, offset=0):
        for i in self.get_requests_cache(self.url_join(self.api_url, f"/playlist/{id}?offset={offset}"))["items"]:
            yield self.song(id=i["track"]["id"], name=i["track"]["name"], artist=i["track"]["artists"][0]["name"])

    def get_playing(self):
        data = self.get_requests_cache(self.url_join(self.api_url, "/current-playing"), max_age=0)
        print(data.keys())
        return self.playing(
            progress=data["progress_ms"] / data["item"]["duration_ms"] * 100,
            track=data["item"]["id"] if "track" in data["item"]["uri"] else None
        )   

    def get_song_features(self, track_id):
        data = self.get_requests_cache(self.url_join(self.api_url, "/recommendations?seed_tracks=" + ",".join(ids)))
        print(data) 

    def url_join(self, a, b):
        if a.endswith("/") and b.startswith("/"):
            b = b[1:]
        return a + b

    """
    Cache related logic.
    """
    def get_requests_cache(self, url, max_age=float('inf')):
        path = self.get_cache(url)
        if os.path.isfile(path) and self.get_file_age_in_seconds(path) < max_age:
            return self._load_cache(path)

        data = requests.get(url, verify=False, headers={
            "Cookie": "auth_token=" + os.getenv("cookie")
        })
        assert data.status_code == 200, f"Internal error, maybe rate liming ? Status code: {data.status_code}, Url: {url}" 
        data = data.json()
        if data is None or data.get("items", True) == None:
            print("Got none from the api, have you added the cookie ?")
            exit(0)
        with open(path, "w") as file:
            file.write(json.dumps(data))
        return data

    def _load_cache(self, path):                        
        with open(path, "r") as file:
            results = json.load(file)
            return results
    
    def get_file_age_in_seconds(self, pathname):
        return time.time() - os.stat(pathname)[stat.ST_MTIME]

    def get_cache(self, url):
        hash = hashlib.sha256(url.encode()).hexdigest()
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
