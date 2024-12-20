import os 
import requests
import hashlib

def _get_id(url):
    url = os.environ["url_to_text_host"] + f"url/id?url={url}"
    print(url)
    try:
        return requests.get(
            url, 
            cookies={
                "credentials": os.environ["auth_header"]
            }
        ).json()["id"]
    except Exception as e:
        print(e)
    return None

def _get_text(id):
    if id is None:
        return None
    try:
        url = os.environ["url_to_text_host"] + f"/text/{id}"
        print(url)
        return requests.get(
            url, 
            cookies={
                "credentials": os.environ["auth_header"]
            }
        ).text
    except Exception as e:
        print(e)
    return None

def get_cache(url):
    path = os.path.join(
        os.path.dirname(__file__),
        ".cache",
        "text-dataset"
    )
    os.makedirs(path, exist_ok=True)
    path = os.path.join(
        path,
        hashlib.sha256(url.encode()).hexdigest()
    )
    return path

def get_text(url):
    cache_path = get_cache(url)
    if os.path.isfile(cache_path):
        with open(cache_path, "r") as file:
            content = file.read()
            if len(content) > 0:
                return content
    id = _get_id(url)
    text = _get_text(id)
    if text is None or len(text) < 100:
        return None
    with open(cache_path, "w") as file:
        file.write(text)
    return text

def get_url_documents():
    url = os.environ["url_to_text_host"] + f"/api/links/1/0"
    items = requests.get(
        url, 
        cookies={
            "credentials": os.environ["auth_header"]
        }
    ).json()
    return [
        get_text(i["url"]) for i in items
    ]
