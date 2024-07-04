import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()

def get_dataset():
    results = None
    if os.path.isfile("cache.json"):
        results = json.load(open("cache.json"))
    else:
        results = requests.get(os.environ["host"], cookies={
            "credentials": os.environ["auth_header"]
        }).json()
        with open("cache.json", "w") as file:
            file.write(json.dumps(results))

    seen = {}
    X = []
    y = []
    for i in results:
        text = i["text"]
        if text in seen:
            continue
        seen[text] = True
        if len(text) > 0:
            X.append(text)
            y.append(i["is_good"])

    return X, y
