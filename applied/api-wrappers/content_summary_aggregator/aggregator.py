import json
import requests
import os 
from dotenv import load_dotenv
from optimization_playground_shared.apis.openai import OpenAiCompletion
from urllib.parse import urlparse
from flask import Flask

load_dotenv()

host = os.environ["real_host"]
print(f"host: {host}")

app = Flask(__name__)

def get_content():
    data = None
    with open("sources.json", "r") as file:
        data = json.load(file)

    entries = []
    for new_source in data["sources"]:
        if not new_source.get("enabled", True):
            continue
        print(new_source["url"])
        html = requests.get(host, params={
            "url": new_source["url"]
        })
        data = html.json()
        urls = []
        summaries = []
        for i in data["urls"]:
            ok = None
            for v in new_source.get("exclude", []):
                if v in i:
                    ok = False
                    break
            if ok == False:
                continue
            for v in new_source["include"]:
                is_okay = v == "<outgoing>" and urlparse(i).hostname != urlparse(new_source["url"])
                if v in i or is_okay:
                    ok = True
                    break
            try:
                if ok:
                    print(f"Fetching {i}")
                    html = requests.get(host, params={
                        "url": i
                    }).json()
                    text = html["text"]
                    if text is None:
                        continue
                    urls.append(i)
                    summaries.append(OpenAiCompletion().get_summary(
                        text
                    ))
            except Exception as e:
                print(e)
        print("\n\n")
        for (url, summary) in zip(urls, summaries):    
            entries.append({
                "url": url,
                "text": summary,
            })
 #           print(summary)
#            print("")
    return entries

entries = get_content()

@app.route('/')
def get_summary():
    template = []
    for v in entries:
        url = v["url"]
        text = v["text"]
        template.append(f"<h1><a href=\"{url}\">{url}</a></h1>")
        template.append(f"<p>{text}</p>")
    template = "\n".join(template)
    return f"""
    <html>
        <body>
            {template}
        </body>
    </html>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=4321)
