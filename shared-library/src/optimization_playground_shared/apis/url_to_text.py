import os 
import requests
import hashlib
import aiohttp
from urllib.parse import urljoin
from ..utils.asyncio_utils import gather_batch
import json

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

def get_text(url, cache_only=False):
    cache_path = get_cache(url)
    if os.path.isfile(cache_path):
        with open(cache_path, "r") as file:
            content = file.read()
            if len(content) > 0:
                return content
    if cache_only:
        return None
    id = _get_id(url)
    text = _get_text(id)
    if text is None or len(text) < 100:
        return None
    with open(cache_path, "w") as file:
        file.write(text)
    return text

def get_url_documents(pages=5):
    documents = []
    for page in range(pages):
        url = os.environ["url_to_text_host"] + f"/api/links/1/{page}"
        items = requests.get(
            url, 
            cookies={
                "credentials": os.environ["auth_header"]
            }
        ).json()
        for i in items:
            text = get_text(i["url"]) 
            if text is None:
                continue
            documents.append(text)
    return documents

"""
async code.
"""
def get_document_eval(n=50):
    host = os.environ["url_to_text_host"]
    urls = urljoin(host, f"/dataset?limit={n}")
    X = []
    y = []
    results = requests.get(urls).json()
    print(results.keys())
    for i in results["entries"]:
        X.append(i["text"])
        y.append(i["is_good"])
    return X, y

async def get_document_dataset():
    host = os.environ["url_to_text_host"]
    urls = [
        urljoin(host, f"/api/dataset"),
        urljoin(host, f"/api/reading_list"),
    ]
    for page in range(1_000):
        urls.append(urljoin(host, f"/api/links/30/{page}"))
    
    operator = lambda x: get_url(x)
    async for v in gather_batch(urls, operator):
        if v is None:
            continue
        documents = json.loads(v)
        document_batch = []
        for v in documents:
            doc_id = v["id"]
            document_batch.append(urljoin(host, f"/text/{doc_id}"))
        async for v in gather_batch(document_batch, operator):
            if v is None:
                continue
            yield v

async def get_url(url):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                results = await response.text()
                if response.status != 200:
                    return None
                return results
    except Exception as e:
        print(e)
        return None
