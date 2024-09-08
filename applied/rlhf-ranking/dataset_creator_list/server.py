from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import json
import hashlib

results_dirname = "results"
os.makedirs(results_dirname, exist_ok=True)

load_dotenv()

app = Flask(__name__)

@dataclass
class Candidate:
    id: str
    url: str
    title: str
    description: str

lookup_db = {}

def get_name(ids):
    ids_name = hashlib.sha256("_".join(sorted(ids)).encode()).hexdigest()
    name = f"{results_dirname}/{ids_name}.json"
    return name

def get_items_to_compare(retry_counter=3):
    global lookup_db
    n = 10
    url = os.environ["host"]
    print(url)
    results = requests.post(url, cookies={
        "credentials": os.environ["auth_header"],
    }, json={
        "n": n,
        "filters": [
            {
                "netloc": "youtube.com"
            }
        ]
    }).json()

    name = get_name(())
    if os.path.isfile(name):
        # Retry ... 
        return get_items_to_compare()
    
    parsed_results = []
    ids = []
    for n in range(1, len(results) + 1):
        lookup_db[int(results[f"candidate_{n}"]["id"])] = results[f"candidate_{n}"]["url"]
        print(lookup_db)
        parsed_results.append(
            Candidate(
                id=results[f"candidate_{n}"]["id"],
                url=results[f"candidate_{n}"]["url"],
                title=results[f"candidate_{n}"]["title"],
                description=results[f"candidate_{n}"]["description"],
            )
        )
        ids.append(results[f"candidate_{n}"]["id"])
    print(ids)
    assert len(ids) == len(list(set(ids))), "Bad ids"
    # Use supervised models to improve speed
    try:
        status = requests.post("http://127.0.0.1:4232/rank", json=list(map(lambda x: x.__dict__, parsed_results)))
        #assert status.status_code == 200, "Bad status code"
        if retry_counter > 0 and status.status_code != 200:
            print(f"Bad status code, retrying ... {retry_counter}")
            return get_items_to_compare(
                retry_counter=retry_counter-1
            )
    except Exception as e:
        print(e)
        print("\t Is the ranking endpoint running ? ")
    item_position = {}
    for index, i in enumerate(status.json()):
        item_position[i["id"]] = index    
    assert len(list(set(item_position.keys()))) == len(list(set(ids)))
    return sorted(parsed_results, key=lambda x: item_position[x.id])

@app.route('/submit_results', methods=["POST"])
def submit_results():
    global lookup_db
    results = request.json["results"]
    print(lookup_db)
    print(results)

    entries = []
    result_ids = []
    for item in results:
        item_id = item["id"]
        score = item["score"]
        item_url = lookup_db[int(item_id)]
        entries.append({
            "id": item_id,
            "url": item_url,
            # Score can be used if all rankings / some are bad 
            "score": score,
        })
        result_ids.append(item_id)
    print(entries)
    name = get_name(result_ids)
    with open(name, "w") as file:
        json.dump(entries, file)
    return jsonify({
        "status": "OK"
    })

@app.route('/')
def index():
    candidates = get_items_to_compare()
    return render_template(
        'index.html', 
        candidates=candidates,
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=2343)
