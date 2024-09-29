import requests
import os
from dotenv import load_dotenv
from langdetect import detect

load_dotenv()

HOST = os.environ["HOST"]


for pages in range(-1, 1_000):
    data = None
    if pages == -1:
        data = requests.get(f"{HOST}/api/dataset", cookies={
            "credentials": os.environ["auth_header"]
        }).json()
    else:
        data = requests.get(f"{HOST}/api/links/90/{pages}", cookies={
            "credentials": os.environ["auth_header"]
        }).json()
    print(pages)
    for i in data:
        text_id = i["id"]
        bucket = text_id % 256
        folder = f"/mnt/blockstorage/text-dataset-2/{bucket}"
        text_file = f"{folder}/{text_id}.txt"
        if os.path.isfile(text_file):
            continue
        response = requests.get(f"{HOST}/text/{text_id}", cookies={
            "credentials": os.environ["auth_header"]
        })
        text = response.text
        if response.status_code != 200 or len(text) == 0:
            continue
        try:
            language = detect(text)
            print((text_id, language))
            if not language in ["no", "en"]:
                continue         
            os.makedirs(folder, exist_ok=True)
            with open(text_file, "w") as file:
                file.write(text)
        except Exception as e:
            print(e)
