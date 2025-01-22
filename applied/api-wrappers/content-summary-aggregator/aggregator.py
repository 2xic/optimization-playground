import json
import requests
from dotenv import load_dotenv
import os 
from optimization_playground_shared.apis.openai import OpenAiCompletion
import time

load_dotenv()

host = os.environ["real_host"]
print(f"host: {host}")

data = None
with open("sources.json", "r") as file:
    data = json.load(file)

for new_source in data["sources"]:
    if not new_source.get("enabled", True):
        continue
    print(new_source["url"])
    html = requests.get(host, params={
        "url": new_source["url"]
    })
    data = html.json()
    summary = []
    for i in data["urls"]:
        ok = None
        for v in new_source["exclude"]:
            if v in i:
                ok = False
                break
        if ok == False:
            continue
        for v in new_source["include"]:
            if v in i:
                ok = True
                break
        if ok:
            html = requests.get(host, params={
                "url": i
            }).json()
            text = html["text"]
            if text is None:
                continue
            summary.append(OpenAiCompletion().get_summary(
                text
            ))
            time.sleep(3)
    text = "\n".join(summary)
    results = OpenAiCompletion().get_summary(text)
    print(results)
