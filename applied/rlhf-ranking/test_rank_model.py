import requests
from dotenv import load_dotenv
load_dotenv()
from optimization_playground_shared.apis.url_to_text import get_text    

def get_documents():
    links = []
    with open(".temp/test_documents", "r") as file:
        for i in file.read().split("\n"):
            if len(i) > 0:
                links.append(i)

    docs = []
    for index, i in enumerate(links):
        docs.append({
            "id": index,
            "text": get_text(i)
        })
    return docs

response = requests.post("http://localhost:2343/", json={
    "documents":get_documents()
})
for i in response.json():
    print(i)
