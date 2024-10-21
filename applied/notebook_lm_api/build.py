import requests
import json
from string import Template
import time
import random
import argparse
import os
from dotenv import load_dotenv
import json

load_dotenv()


parser = argparse.ArgumentParser(description="Process a URL.")
parser.add_argument("podcast_url", help="The URL to be processed")
parser.add_argument("name", help="The name of the notebook")
kind = "url_kind"

args = parser.parse_args()
podcast_url = args.podcast_url

config = None
with open("config.json" ,"r", encoding="utf-8") as file:
    config = json.load(file)

def escape_quotes_and_backslashes(text):
    return text.replace('\\', '\\\\').replace('"', '\\"')


if os.environ["raw_host"] in podcast_url:
    kind = "text_kind"
    podcast_url = requests.get(podcast_url, cookies={
        "credentials": os.environ["auth_header"]
    }).text
    # safley stringify.
    podcast_url = json.dumps(podcast_url)[1:-1]

# create the entry
urls = config["urls"]
cookie = config["cookie"]
headers = {
    "Host": "notebooklm.google.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:131.0) Gecko/20100101 Firefox/131.0",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "gzip, deflate, br, zstd",
    "Referer": "https://notebooklm.google.com/",
    "X-Same-Domain": "1",
    "Content-Type": "application/x-www-form-urlencoded;charset=utf-8",
    "Origin": "https://notebooklm.google.com",
    "Alt-Used": "notebooklm.google.com",
    "Connection": "keep-alive",
    "Cookie": cookie,
    "Sec-Fetch-Dest": "empty",
    "Sec-Fetch-Mode": "cors",
    "Sec-Fetch-Site": "same-origin",
    "Priority": "u=0",
    "TE": "trailers",
}

if True:
    modified_payload= urls[0]["payload"]
    modified_payload["f.req"] = Template(
        modified_payload["f.req"]
    ).substitute({
        "name": args.name
    })

    data = requests.post(
        urls[0]["url"],
        data=modified_payload,
        headers=headers
    )

    #print(data.text)
    assert data.status_code == 200

    respone = json.loads(data.text.split("\n")[3])
    respone_id = json.loads(respone[0][2])[2]

    time.sleep(2)
    print(
        respone_id
    )
    print(data.status_code)


"""
Send in the url
"""
#print(podcast_url)
modified_payload= urls[1]["options"][kind]["payload"]
modified_payload["f.req"] = Template(
    modified_payload["f.req"]
).substitute({
    "id": respone_id,
    "url": podcast_url,
})
#print(podcast_url)
#exit(0)

data = requests.post(
    urls[1]["url"],
    data=modified_payload,
    headers=headers,
)
print(data.text)
print(data.status_code)
assert data.status_code == 200
time.sleep(random.randint(1, 3))


#print(modified_payload)
#exit(0)

"""
Start the actual text generation
"""

modified_payload= urls[2]["payload"]
modified_payload["f.req"] = Template(
    modified_payload["f.req"]
).substitute({
    "id": respone_id,
})

data = requests.post(
    urls[2]["url"],
    data=modified_payload,
    headers=headers,
)
print(data.text)
print(data.status_code)
assert data.status_code == 200
