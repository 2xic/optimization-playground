"""
november 2024 - note: sadly the api is already changed so this doesn't work anymore.
april 2025 - updated it again

Most payloads are hidden to prevent misuse. Please be kind. 

NO WARRANTY GIVEN, BE KIND.
"""
import requests
import json
import argparse
from dotenv import load_dotenv
import json
from config import payloads
import urllib.parse

load_dotenv()

parser = argparse.ArgumentParser(description="Process a URL.")
parser.add_argument("url", help="The URL to be processed")
parser.add_argument("name", help="The name of the notebook")
kind = "url_kind"

args = parser.parse_args()
url = args.url

config = payloads

def escape_quotes_and_backslashes(text):
    return text.replace('\\', '\\\\').replace('"', '\\"')

# create the entry
urls = config["urls"]
cookie = config["cookie"]
at = config["at"]

headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:136.0) Gecko/20100101 Firefox/136.0', 
    'Accept': '*/*', 
    'Accept-Language': 'en-US,en;q=0.5', 
    'Accept-Encoding': 'gzip, deflate, br, zstd', 
    'Referer': 'https://notebooklm.google.com/', 
    'X-Same-Domain': '1', 
    'Content-Type': 'application/x-www-form-urlencoded;charset=utf-8', 
    'Origin': 'https://notebooklm.google.com', 
    'Alt-Used': 'notebooklm.google.com', 
    'Connection': 'keep-alive', 
    'Cookie': cookie,
    'Sec-Fetch-Dest': 'empty', 
    'Sec-Fetch-Mode': 'cors', 
    'Sec-Fetch-Site': 'same-origin', 
    'Priority': 'u=0' 
}


def cache_response(response: requests.Response, name: str):
    with open(f"{name}.json", "wb") as file:
        file.write(response.content)
    return response

def load_cache(name):
    with open(f"{name}.json", "rb") as file:
        return file.read()
    
def get_request_body(content):
    data = json.loads(content.split(b"\n")[3])[0][2]
    return json.loads(data)

def get_notebook_id_from_content(data):
    data = get_request_body(data)[2]
    return data 

def get_source_ids_for_audio(data, notebook_id):
    data = get_request_body(data)
    for i in data[0]:
        res = i[1]
        if type(res) == list and notebook_id == i[2]:
            return res[0][2][3][0]
    raise Exception("Failed to find source id")

def get_source_id_from_add(data):
    data = get_request_body(data)
    return data[0][0][0][0]
    
# print(get_source_id_from_add(load_cache("add_source")))
# exit(0)


def get_notebook_id():
    creation_payload = urls[0]["payload"]
    creation_payload["f.req"] = '[[["CCqFvf","[\\"\\"]",None,"generic"]]]'
    creation_payload["at"] = at
    encoded_payload = urllib.parse.urlencode(creation_payload) + "&"

    url = urls[0]["url"]
    data = cache_response(
        requests.post(
            url,
            data=encoded_payload,
            headers=headers
        ),
        "notebook_id"
    )

    return get_notebook_id_from_content(data.content)

def add_source(notebook_id, url):
    payload = urllib.parse.urlencode({
        "f.req": f'[[["izAoDd","[[[null,null,[\\"{url}\\"],null,null,null,null,null,null,null,1]],\\"{notebook_id}\\"]",null,"generic"]]]',
        "at": at,
    }) + "&"
    url = urls[1]["url"]

    data = cache_response(
        requests.post(
            url,
            data=payload,
            headers=headers
        ),
        "add_source"
    )
    print(data.status_code)
    print(data.text)
    return get_source_id_from_add(data.content)

def get_document_source_ids(notebook_id):
    payload = urllib.parse.urlencode({
        "f.req": f'[[["wXbhsf","[null,1]",null,"generic"]]]',
        "at": at,
    }) + "&"
    url = urls[3]["url"]
    data = cache_response(
        requests.post(
            url,
            data=payload,
            headers=headers
        ),
        "get_document_source_ids"
    )
    print(data.status_code)
    print(data.text)
    return get_source_ids_for_audio(data.content, notebook_id)

def generate_audio(notebook_id, source_id):
    payload = urllib.parse.urlencode({
        "f.req": f'[[["AHyHrd","[\\"{notebook_id}\\",0,[null,null,null,[[\\"{source_id}\\"]]]]",null,"generic"]]]',
        "at": at,
    }) + "&"
    url = urls[2]["url"]
    data = cache_response(
        requests.post(
            url,
            data=payload,
            headers=headers
        ),
        "generate_audio"
    )
    print(data.status_code)
    print(data.text)

if __name__ == "__main__":
    notebook_id = get_notebook_id()
    source_id = add_source(
        notebook_id,
        "https://en.wikipedia.org/wiki/Robot_control"
    )   
    generate_audio(notebook_id, source_id)


