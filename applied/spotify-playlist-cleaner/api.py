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
load_dotenv()

def get_cache(url):
    hash = hashlib.sha256(url.encode()).hexdigest()[:8]
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        ".cache",
        hash
    )
    return path


def play_song(song_id):
    data = requests.post(f"https://localhost:8089/play/{song_id}", verify=False, headers={
        "Cookie": "auth_token=" + os.getenv("cookie")
    })
    print(data)

def get_requests_cache(url):
    path = get_cache(url)
    if os.path.isfile(path):
        print(path)
        with open(path, "r") as file:
            return json.loads(file.read())
    print(os.getenv("cookie"))
    data = requests.get(url, verify=False, headers={
        "Cookie": "auth_token=" + os.getenv("cookie")
    }).json()
    time.sleep(1)
    with open(path, "w") as file:
        file.write(json.dumps(data))
    return data

playlist = namedtuple('playlist', ['id', 'name'])
song = namedtuple('song', ['id', 'name', 'image'])
feature = namedtuple('features', [
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

def get_playlists():
    for i in get_requests_cache("https://localhost:8089/playlists")["items"]:
        if i["owner"]["display_name"] != "brage":
            continue
        yield playlist(id=i["id"], name=i["name"])

def get_playlist_songs(id, offset=0):
    print(f"https://localhost:8089/playlist/{id}?offset={offset}")
    for i in get_requests_cache(f"https://localhost:8089/playlist/{id}?offset={offset}")["items"]:
        url = (i["track"]["album"]["images"][-1]["url"])
        yield song(id=i["track"]["id"], name=i["track"]["name"], image=url)

def get_song_feature(id):
    for i in get_requests_cache(f"https://localhost:8089/song/{id}/feature")["audio_features"]:
        yield feature(
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
