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

api_url = "https://localhost:8089"
one_hour = 60 * 60
playlist_max_age = 12 * one_hour
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

def get_cache(url):
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

def play_song(song_id):
    data = requests.post(f"{api_url}/play/{song_id}", verify=False, headers={
        "Cookie": "auth_token=" + os.getenv("cookie")
    })
    print(data)

def move_song(song_id, from_playlist, to_playlist):
    data = requests.post(f"{api_url}/playlist/move/{from_playlist}/{to_playlist}/{song_id}", verify=False, headers={
        "Cookie": "auth_token=" + os.getenv("cookie")
    })
    print(data)

def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]

def get_requests_cache(url, max_age=float('inf')):
    path = get_cache(url)
    if os.path.isfile(path):
        if file_age_in_seconds(path) < max_age:
            with open(path, "r") as file:
                content = file.read()
                if len(content):
                    results = json.loads(content)
                    print(results)
                    print(results.get("items", True) != None)
                    if results.get("items", True) != None:
                        return results
    data = requests.get(url, verify=False, headers={
        "Cookie": "auth_token=" + os.getenv("cookie")
    }).json()
    print(data)
    if data is None or data.get("items", True) == None:
        print("Got none from the api, have you added the cookie ?")
        exit(0)
    time.sleep(1)
    with open(path, "w") as file:
        file.write(json.dumps(data))
    return data

def get_playlists():
    for i in get_requests_cache(f"{api_url}/playlists")["items"]:
        if i["owner"]["display_name"] != "brage":
            continue
        yield playlist(id=i["id"], name=i["name"])

def get_playlist_songs(id, offset=0):
    print(f"{api_url}/playlist/{id}?offset={offset}")
    for i in get_requests_cache(f"{api_url}/playlist/{id}?offset={offset}", max_age=playlist_max_age)["items"]:
        url = (i["track"]["album"]["images"][-1]["url"])
        yield song(id=i["track"]["id"], name=i["track"]["name"], image=url)

def get_song_feature(id):
    for i in get_requests_cache(f"{api_url}/song/{id}/feature")["audio_features"]:
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
