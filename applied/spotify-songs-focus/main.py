"""
Spotify has some apis to solve this
    - https://developer.spotify.com/documentation/web-api/reference/get-recommendations
    - https://developer.spotify.com/documentation/web-api/reference/get-the-users-currently-playing-track
    - https://developer.spotify.com/documentation/web-api/reference/add-to-queue
    - https://developer.spotify.com/documentation/web-api/reference/get-users-top-artists-and-tracks

Flow would be something like
    - We listen for changes in playing music (+1 for completing song, -1 for not completing song)

Seed (?)
    - Need to provide one input to start the seed from, I think. Else we have a cold start problem.
    - Probably need to crawl some playlists etc. to get some features/recommendations also.

"""

import argparse
import time
from api import Api
import json

def get_playlist_id(id_or_url):
    return id_or_url.replace("https://open.spotify.com/playlist/", "").split("?")[0]

def main():
    api = Api()
    parser = argparse.ArgumentParser(description="Process a playlist ID.")
    parser.add_argument(
        "playlist_id", 
        type=str, 
        help="The ID of the playlist to process."
    )
    
    args = parser.parse_args()
    playlist_id = get_playlist_id(args.playlist_id)
    print(f"Processing playlist ID: {playlist_id}")
    # This is the seed
    track_ids = []
    for i in api.get_playlist(playlist_id):
        track_ids.append(i.id)
    good = set()
    bad = set()
    last_song = api.get_playing()
    """
    This generates all the data we need
    """
    while True:
        new_song = api.get_playing()
        print((
            last_song.track,
            new_song.track
        ))
        # percentage
        if last_song.progress > 90:# and last_song.track != new_song.track:
            good.add(last_song.track)
        elif last_song.track != new_song.track:
            bad.add(last_song.track)
        last_song = new_song

        print(json.dumps({
            "good": list(good),
            "bad": list(bad)
        }, indent=4))
        time.sleep(5)        
    
    """
    Then that should go into some feature extractor to get new recommendations and such
    """

if __name__ == "__main__":
    main()
