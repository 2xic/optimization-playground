"""
Consumers get a simple JSON requests with a predefined format from the 
producer
"""
from flask import Flask, request, jsonify
import os
from .metrics import Metrics
import json
import dataclasses
import glob
from.plot import plot_xy
import base64

root_data_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".data"
)
global_metadata = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".data",
    "metadata.json"
)

class SimpleDatabase:
    def __init__(self, project_name, id) -> None:
        self.project_metadata_path = os.path.join(
            root_data_directory,
            project_name,
            "metadata.json",
        )
        self.project_path = os.path.join(
            root_data_directory,
            project_name,
            id
        )
        os.makedirs(self.project_path, exist_ok=True)
        self.run_id = id
        self.project_name = project_name

    def store_config(self, timestamp):
        current_global_config = {}
        current_project_config = {}
        if os.path.isfile(self.project_metadata_path):
            with open(self.project_metadata_path, "r") as file:
                current_project_config = json.load(file)
        if os.path.isfile(global_metadata):
            with open(global_metadata, "r") as file:
                current_global_config = json.load(file)

        current_project_config[self.run_id] = {
            "last_commit": timestamp
        }
        current_global_config[self.project_name] = {
            "last_commit": timestamp
        }
        with open(self.project_metadata_path, "w") as file:
            json.dump(current_project_config, file)
        with open(global_metadata, "w") as file:
            json.dump(current_global_config, file)

    def add_metric(self, data):
        metric = Metrics(**data)
        epoch = str(metric.epoch)
        file_path = os.path.join(
            self.project_path,
            epoch + ".json"
        )
        with open(file_path, "w") as file:
            json.dump(dataclasses.asdict(metric), file)
        self.store_config(metric.timestamp)


app = Flask(__name__)
@app.route('/', methods=['POST'])
def add_message():
    """
    Takes the logging metadata and stores it
    """
    content = request.json
    print(content)
    metrics = SimpleDatabase(content["name"], content["run_id"]).add_metric(
        content["metrics"]
    )
    print(metrics)
    return jsonify({
        "status": "success"
    })

@app.route('/', methods=['GET'])
def index():
    """
    Shows the projects data
    """
    with open(global_metadata, "r") as file:
        data = json.load(file)
        results = []
        for (name, last_commit) in sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True):
            print(name)
            results.append(
                f"<a href={name}>{name}</a>".format(name=name)
            )
        return "<br>".join(results)
    
@app.route('/<project_name>', methods=['GET'])
def runs(project_name):
    """
    Shows the project run data
    """
    project_metadata_path = os.path.join(
        root_data_directory,
        project_name,
        "metadata.json",
    )
    with open(project_metadata_path, "r") as file:
        data = json.load(file)
        results = []
        for (name, last_commit) in sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True):
            print(name)
            results.append(
                f"<a href={project_name}/{name}>{name}</a>".format(project_name=project_name,name=name)
            )
        return "<br>".join(results)

@app.route('/<project_name>/<run_id>', methods=['GET'])
def run(project_name, run_id):
    """
    Shows the project run data
    """
    run_metadata_path = os.path.join(
        root_data_directory,
        project_name,
        run_id,
        "*.json"
    )
    data = []
    for i in glob.glob(run_metadata_path):
        with open(i, "r") as file:
            data.append(json.load(file))
    data = sorted(data, key=lambda x: x["epoch"])
    loss_plot = plot_xy(
        list(map(lambda x: x["loss"], data))
    )
    accuracy_plot = plot_xy(
        list(map(lambda x: x["training_accuracy"], data))
    )
    predictions = list(map(lambda x: get_prediction_format(x), data))[-5:]
    return "<br>".join([
        "<h2>Loss</h2>",
        loss_plot,
        "<h2>Accuracy</h2>",
        accuracy_plot,
        "<br>".join(predictions),
        "<h1>Metrics</h1>",
    ])

def get_prediction_format(entry):
    value = [
        "<h3>{epoch}</h3>".format(epoch=entry["epoch"])
    ]
    if entry["prediction"]["prediction_type"] == "text":
        value.append(entry["prediction"]["value"].replace("\n", "<br>"))
    elif entry["prediction"]["prediction_type"] == "image":
        data = base64.b64encode(
            # slice the 0x
            bytes.fromhex(entry["prediction"]["value"][2:])
        ).decode("ascii")
        value.append(f"<img src='data:image/png;base64,{data}'/>")
    return "<br>".join(value)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=8181)
