from dotenv import load_dotenv
import requests
import os
load_dotenv()

def get_tweets():
    tweets = requests.get(
        os.environ.get("HOST")
    ).json()
    return list(map(lambda x: x["tweet"]["text"], tweets)),\
            list(map(lambda x: x["tweet"].get("user", {"name":"???"})["name"], tweets))

print(get_tweets())
