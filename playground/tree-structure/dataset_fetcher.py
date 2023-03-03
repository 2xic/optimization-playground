import requests
from dotenv import load_dotenv
import os
load_dotenv()
import time

reesponse = requests.get(os.environ["HOST"]).json()
for i in reesponse:
    id = i["id"]
    url = i["url"]
    score = str(i["score"])
    
    print(url)

    path = os.path.join('dataset', score, f"{id}.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    os.system(f"/snap/bin/chromium --headless --disable-gpu --screenshot={path} {url}")
    time.sleep(1)
