from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv
from dataclasses import dataclass
import json

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

def get_name(candidate_1_id, candidate_2_id):
    [first, second] = sorted([candidate_1_id, candidate_2_id])
    name = f"{results_dirname}/{first}_{second}.json"
    return name

def get_items_to_compare():
    global lookup_db
    url = os.environ["host"]
    results = requests.get(url, cookies={
        "credentials": os.environ["auth_header"]
    }).json()
    # This should return useful results, yes.
    # url, title and some text.
    lookup_db[results["candidate_1"]["id"]] = results["candidate_1"]["url"]
    lookup_db[results["candidate_2"]["id"]] = results["candidate_2"]["url"]

    name = get_name(results["candidate_1"]["id"], results["candidate_2"]["id"])
    if os.path.isfile(name):
        # Retry ... 
        return get_items_to_compare()
    
    return (
        Candidate(
            id=results["candidate_1"]["id"],
            url=results["candidate_1"]["url"],
            title=results["candidate_1"]["title"],
            description=results["candidate_1"]["description"],
        ),
        Candidate(
            id=results["candidate_2"]["id"],
            url=results["candidate_2"]["url"],
            title=results["candidate_2"]["title"],
            description=results["candidate_2"]["description"],
        )
    )
 
@app.route('/submit_results', methods=["POST"])
def submit_results():
    global lookup_db
    # And return the new items
    data = request.json["results"]
    print(lookup_db)
    print(data)
    winner_id = data["winner_id"]
    winner_url = lookup_db[winner_id]

    looser_id = data["looser_id"]
    looser_url = lookup_db[looser_id]

    name = get_name(winner_id, looser_id)
    with open(name, "w") as file:
        json.dump({
            "winner": {
                "winner_id": winner_id,
                "winner_url": winner_url,
            },
            "looser": {
                "looser_id": looser_id,
                "looser_url": looser_url,
            }
        }, file)
    return jsonify({
        "status": "OK"
    })

@app.route('/')
def index():
    (candidate_1, candidate_2) = get_items_to_compare()
    return render_template(
        'index.html', 
        candidate_1=candidate_1,
        candidate_2=candidate_2,
    )

if __name__ == "__main__":
    app.run(host='0.0.0.0')
