import requests
import os
from dotenv import load_dotenv
import time
import json
import requests
from urllib3.exceptions import InsecureRequestWarning

# Suppress only the single warning from urllib3 needed.
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

load_dotenv()

# just looking at the spotify request
def get_lyrics(track_id):
    try:
        response = requests.get(
            f"https://spclient.wg.spotify.com/color-lyrics/v2/track/{track_id}?format=json&vocalRemoval=false&market=from_token",
            headers={
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) Gecko/20100101 Firefox/125.0',
                'Accept': 'application/json',
                'Accept-Language': 'en',
                'Accept-Encoding': 'gzip, deflate, br',
                'Referer': 'https://open.spotify.com/',
                'app-platform': 'WebPlayer',
                'spotify-app-version': '1.2.37.562.gac73a9fe',
                'client-token':  os.environ["client_token"],
                'Origin': 'https://open.spotify.com',
                'Sec-Fetch-Dest': 'empty',
                'Sec-Fetch-Mode': 'cors',
                'Sec-Fetch-Site': 'same-site',
                'authorization': os.environ["authorization"], 
                'Connection': 'keep-alive',
            }
        )
        assert response.status_code == 200        
        return response.json()
    except Exception as e:
        print(e)
        return None

## using the wirehead api
def get_playlist(playlist_id, offset=0):
    host = os.environ["host"]
    #print(f"{host}/playlist/{playlist_id}")
    response = requests.get(
        f"{host}/playlist/{playlist_id}?offset={offset}",
        headers={
            "Cookie":os.environ["spotify_api"],
        },
        verify=False
    )
    return response.json()

if __name__ == "__main__":
    playlist_id = os.environ["playlist_id"]
    offset = 0
    crawled = 0
    playlist_entries = []
    while crawled <= 10:
        items = get_playlist(playlist_id, offset)["items"]
      #  print(items)
        for i in items:
            track_id = i["track"]["id"]        
            track_name = i["track"]["name"]

            file = f".lyrics/{track_id}.json"
            if os.path.isfile(file):
                playlist_entries.append(track_id)
                continue

            track_lyrics = get_lyrics(track_id)

            with open(file, "w") as file:
                file.write(json.dumps({
                    "name": track_name,
                    "lyrics": track_lyrics,
                }))

            time.sleep(5)
            crawled += 1
        offset += len(items)
        print(f"offset: {offset}")
        if len(items) == 0:
            print("DONE")
            break
    with open(f"playlist_{playlist_id}.json", "w") as file:
        file.write(json.dumps(playlist_entries))
