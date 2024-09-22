import requests
import os
from dotenv import load_dotenv

load_dotenv()

HOST = os.environ["HOST"]


for pages in range(0, 1_000):
    data = requests.get(f"{HOST}/api/links/90/{pages}", cookies={
        "credentials": os.environ["auth_header"]
    }).json()
    print(pages)
    for i in data:
        text_id = i["id"]
        bucket = text_id % 256
        folder = f"/mnt/blockstorage/text-dataset/{bucket}"
        os.makedirs(folder, exist_ok=True)
        text_file = f"{folder}/{text_id}.txt"
        if os.path.isfile(text_file):
            continue
        response = requests.get(f"{HOST}/text/{text_id}", cookies={
            "credentials": os.environ["auth_header"]
        })
        text = response.text
        if response.status_code != 200 or len(text) == 0:
            continue
        with open(text_file, "w") as file:
            file.write(text)
