import requests
from dotenv import load_dotenv
import os
import json
import hashlib
import time

os.makedirs(".cache", exist_ok=True)

if os.path.isfile(".env"):
    load_dotenv()
else:
    load_dotenv(".env.minimal")

def get_dataset(force=False):
    all_results = None
    if os.path.isfile("training_data.json") and not force:
        all_results = json.load(open("training_data.json"))
    else:
        all_results = []
        limit = 100
        offset = 0
        while True:
            print("Downloading ... offset = ", offset)
            url = os.environ["host"] + f"?offset={offset}&limit={limit}"
            path = os.path.join(".cache", hashlib.sha256(url.encode()).hexdigest())
            results = None

            if os.path.isfile(path):
                with open(path, "rb") as file:
                    results = json.load(file)
            else:
                start = time.time()
                results = requests.get(url, cookies={
                    "credentials": os.environ["auth_header"]
                }).json()
                end = time.time()
                with open(path, "w") as file:
                    json.dump(results, file)
                if (end - start) > 15:
                    print("Long response ... early exiting")
                    break
            all_results += results["entries"]
            offset = results["next_offset"]
            print((len(results["entries"]), len(all_results), results["next_offset"], results["done"]))
            if results["done"]:
                break
            elif len(all_results) > 1_000:
                print("early breaking")
                break
        with open("cache.json", "w") as file:
            file.write(json.dumps(all_results))

    seen = {}
    X = []
    y = []
    for i in all_results:
        text = i["text"]
        if text in seen:
            continue
        seen[text] = True
        if len(text.strip()) > 0:
            X.append(text)
            y.append(i["is_good_content"])
    return X, y

if __name__ == "__main__":
    get_dataset(force=True)
