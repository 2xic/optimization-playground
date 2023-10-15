"""
Copied from https://github.com/2xic-archive/arxiv-citation-graph/tree/master
"""
import random
import time
import urllib.request
import os
import glob
import re
from typing import List

class Document:
    def __init__(self, id, path, text) -> None:
        self.id = id
        self.path = path
        self.text = text

def get_download_path(name):
    if "/" in name:
        return name
    root = os.path.join(
        os.path.dirname(__file__),
        ".cache",
    )
    os.makedirs(root, exist_ok=True)
    return os.path.join(
        root,
        name
    )

def download_raw(name):
    path = get_download_path("{}.pdf".format(name))
    if not os.path.isfile(path):
        url = "https://arxiv.org/pdf/{}.pdf".format(name)
        print("Downloading {}".format(url))
        urllib.request.urlretrieve(
            url, path)
        return path
    return path


def download_process(name):
    doc_name = download_raw(name)
    if not doc_name is None and not os.path.isfile(doc_name.replace(".pdf", ".txt")):
        document_text_path = doc_name.replace(".pdf", ".txt")
        os.system("pdftotext {} {}".format(
            doc_name, document_text_path))
        # a nice crawler that takes a break
        time.sleep(random.randint(2, 5))
        if os.path.isfile(document_text_path):
            with open(document_text_path, "r") as file:
                return Document(
                    name,
                    document_text_path,
                    file.read()
                )
    # already processed
    return None

def return_citation(doc):
    doc_name = "{}.txt".format(doc) if not ".txt" in doc else doc
    text = open(get_download_path(doc_name), "r").read()
    citation = re.findall(r"arXiv:\d*.\d*", text)
    for i in citation:
        yield i.replace("arXiv:", "")


def crawl_some(limit_downloads=100) -> List[Document]:
    downloads = 0
    outputs = []
    while downloads < limit_downloads:
        for docs in glob.glob(get_download_path("*.txt")):
            for citation in return_citation(docs):
                processed = download_process(citation)
                # already downloaded
                if processed is None:
                    continue
                outputs.append(processed)
                downloads += 1
                if limit_downloads < downloads:
                    break
        downloads += 1
    return outputs

def get_crawled():
    items = []
    for document_text_path in glob.glob(get_download_path("*.txt")):
        with open(document_text_path, "r") as file:
            item = Document(
                os.path.basename(document_text_path).replace(".txt", ""),
                document_text_path,
                file.read()
            )
            items.append(item)
    return items

def start_crawling(seed, limit_downloads=1_00) -> List[Document]:
    for i in seed:
        download_process(i)

    return crawl_some(limit_downloads=limit_downloads)
