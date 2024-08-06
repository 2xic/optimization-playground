from bs4 import BeautifulSoup
import hashlib
import os
import requests
import time
from tqdm import tqdm
import json
from dataclasses import dataclass

@dataclass
class Artists:
    name: str
    image: str
    score: float
    best_track: str

def get_hash_path(url):
    hash_id = hashlib.sha256(url.encode()).hexdigest()
    return os.path.join(
        ".cache",
        hash_id
    )

def requests_with_cache(url):
    path = get_hash_path(url)
    dir_name = os.path.dirname(path)
    os.makedirs(dir_name, exist_ok=True)

    if os.path.isfile(path):
        with open(path, "r") as file:
            return file.read()
    data = requests.get(url)
    with open(path, "w") as file:
        file.write(data.text)
    time.sleep(2)
    return data.text

def get_artist_scores():
    spotify_playlist_cleaner_endpoint = "http://localhost:5000"
    main_embed = "https://embeds.appmiral.com/d7nDnUtMvU/en/embed.html?version=1722940803"
    text = requests_with_cache(main_embed)
    artists = BeautifulSoup(text).find_all("div",onclick=lambda x: x is not None and "appm_showDetail2" in x)

    artists_score = []
    for i in tqdm(artists):
        artist_name = (i.attrs.get("data-artist-name", None))
        if artist_name is None:
            continue
        artist_id = i.find("img", onclick=True)
        if artist_id is None:
            continue
        artist_id = artist_id.attrs["data-id"]
        artist_info = f"https://embeds.appmiral.com/d7nDnUtMvU/en/detail/artist-{artist_id}.html"
        page = BeautifulSoup(requests_with_cache(artist_info))
        spotify_artists_url = page.find("a", href=lambda x: "https://open.spotify.com/artist/" in x)
        if spotify_artists_url is None:
            continue
        spotify_artists_url = spotify_artists_url["href"]
        spotify_artists_id = spotify_artists_url.replace("https://open.spotify.com/artist/", "")

        print(artist_name)
        print(f"\t{artist_id}")
        print(f"\t{artist_info}")
        print(f"\t{spotify_artists_url}")
        # Now we fetch it on the backend by requesting the playlist link.
        # We train a model on that data and score it based on my other music.
        # 
        response = requests.get(f"{spotify_playlist_cleaner_endpoint}/artists/top_tracks?artists_id={spotify_artists_id}").json()
        if response is None:
            continue
        # THen we need to get the songs features 
        total_score = 0
        for i in response["tracks"]:
            track_id = i["id"]
            song_features = requests.get(f"{spotify_playlist_cleaner_endpoint}/song/distance?song_id={track_id}").json()
            total_score += song_features["distance"]
        #print((artist_name, total_score))
        artists_score.append(
            Artists(
                name=artist_name,
                score=total_score,
                image=None,
                best_track=None
            )
        )
    return sorted(artists_score, key=lambda x: x.score)[:10]

if __name__ == "__main__":
    get_artist_scores()
