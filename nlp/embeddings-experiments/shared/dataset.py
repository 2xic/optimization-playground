import requests
import json
import os

def create_dataset(count=1_000):
    tweets = [

    ]
    HOST = os.getenv("HOST")
    for i in range(0, count, 50):
        url = f"http://{HOST}:8081/?first={i}"
        for i in (requests.get(url).json()):
            tweets.append(i["tweet"]["text"])
    return tweets
