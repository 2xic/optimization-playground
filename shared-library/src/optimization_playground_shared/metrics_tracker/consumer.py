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
from ..plot.Plot import Plot, Figure
from typing import Dict
import difflib

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
        self.project_metadata_path = SimpleDatabase.metadata_file(project_name)
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

    def add_code_state(self, files: Dict[str, str]):
        file_path = self._get_diff_path(self.run_id)
        if not os.path.isfile(file_path):
            with open(file_path, "w") as file:
                file.write(json.dumps(files))
            print("wrote to", file_path)
        # ^ does not generate a diff, but stores the file state

    def get_code_diff(self, ref_2, context=False):        
        ref_1 = self.load_diff(self.run_id)
        ref_2 = self.load_diff(ref_2)

        if ref_1 is None or ref_2 is None:
            return "One of the files / Both files don't have the state file"

        diff = []
        for key, value in ref_1.items():
            reference_file_content = ref_2.get(key, None)
            if reference_file_content is None:
                reference_file_content = ""
            code_diff = difflib.HtmlDiff().make_file(
                value.split("\n"),
                reference_file_content.split("\n"),
                key,
                key,
                context=context
            )
            # -> We want a nicer change :(
            if "No Differences Found" in code_diff:
                continue
            diff.append(
                code_diff
            )

        return "<br>".join(diff)

    def load_diff(self, run_id):
        file_path = self._get_diff_path(run_id)
        print(file_path)
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                return json.loads(file.read())
        return None
        
    def _get_diff_path(self, run_id):
        return os.path.join(
            root_data_directory,
            self.project_name,
            run_id,
            "file_state.json"
        )

    @staticmethod
    def get_sorted_runs(project_name):
        project_metadata_path = SimpleDatabase.metadata_file(project_name)
        with open(project_metadata_path, "r") as file:
            data = json.load(file)
            ids = []
            for (run_id, _) in sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True):
                ids.append(run_id)
            return ids
        
    @staticmethod
    def metadata_file(project_name):
        return os.path.join(
            root_data_directory,
            project_name,
            "metadata.json",
        )

app = Flask(__name__)
@app.route('/', methods=['POST'])
def add_message():
    """
    Takes the logging metadata and stores it
    """
    content = request.json
    print(content)
    message_type = content.get("message_type", None)
    if message_type == "metrics":
        metrics = SimpleDatabase(content["name"], content["run_id"]).add_metric(
            content["metrics"]
        )
        print(metrics)
    elif message_type == "code_state":
        metrics = SimpleDatabase(content["name"], content["run_id"]).add_code_state(
            content["code"]
        )
        print(metrics)
    else:
        return jsonify({
            "status": "error",
            "message": f"Unknown {message_type}"
        })

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
        for (name, _) in sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True):
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
    results = []
    ids = SimpleDatabase.get_sorted_runs(project_name)
    for index, run_id in enumerate(ids):
        if (index + 1) < len(ids):
            next_id = ids[index + 1]
            results.append(
                f"<a href={project_name}/{run_id}>{run_id}</a> (<a href={project_name}/compare/{next_id}/{run_id}>diff</a>)".format(project_name=project_name, run_id=run_id, next_id=next_id)
            )
        else:
            results.append(
                f"<a href={project_name}/{run_id}>{run_id}</a>".format(project_name=project_name, run_id=run_id)
            )
    return "<br>".join(results)

@app.route('/<project_name>/compare/<ref_1>/<ref_2>', methods=['GET'])
def diff_runs(project_name, ref_1, ref_2):
    full_link = f'/{project_name}/compare/{ref_1}/{ref_2}/full'
    full_link = f"<a href={full_link}>full diff</a>"
    return full_link + SimpleDatabase(project_name, ref_1).get_code_diff(
        ref_2,
        context=True
    )

@app.route('/<project_name>/compare/latest', methods=['GET'])
def diff_runs_latest(project_name):
    ids = SimpleDatabase.get_sorted_runs(project_name)
    ref_1 = ids[1]
    ref_2 = ids[0]
    full_link = f'/{project_name}/compare/{ref_1}/{ref_2}/full'
    full_link = f"<a href={full_link}>full diff</a>"
    return full_link + SimpleDatabase(project_name, ref_1).get_code_diff(
        ref_2,
        context=True
    )

@app.route('/<project_name>/compare/<ref_1>/<ref_2>/full', methods=['GET'])
def diff_runs_full(project_name, ref_1, ref_2):
    return SimpleDatabase(project_name, ref_1).get_code_diff(
        ref_2,
        context=False
    )

def load_run_id(project_name, run_id):
    run_metadata_path = os.path.join(
        root_data_directory,
        project_name,
        run_id,
        "*.json"
    )
    data = []
    for i in glob.glob(run_metadata_path):
        if os.path.basename(i).split(".")[0].isnumeric():
            with open(i, "r") as file:
                data.append(json.load(file))
    if len(data) == 0:
        return None
    data = sorted(data, key=lambda x: x["epoch"])
    loss_plot = []
    if all(list(map(lambda x: type(x["loss"]) == dict, data))):
        plots = {}
        for i in data:
            for key, value in i["loss"].items():
                if key in plots:
                    plots[key].append(value)
                else:
                    plots[key] = [value]
        all_plots = [
            plots,
        ]
        for key, value in plots.items():
            all_plots.append({key: value})
        for current_plot in all_plots:
            plot = Plot()
            path = plot.plot_figures(
                figures=[
                    Figure(
                        plots=current_plot,
                        title="Loss",
                        x_axes_text="Epochs",
                        y_axes_text="Loss",
                    ),
                ],
                name='.training.png'
            )
            with open(path, "rb") as file:
                encoded_data = base64.b64encode(file.read()).decode("ascii")
                loss_plot.append( f"<img src='data:image/png;base64,{encoded_data}'/>")
    elif  all(list(map(lambda x: type(x["loss"]) in [float, int], data))):
        loss_plot.append(plot_xy(
            list(filter(lambda x: x is not None, list(map(lambda x: x["loss"], data))))
        ))
    accuracy = ""
    if not all(list(map(lambda x: x["training_accuracy"] is None, data))):
        accuracy_plot = plot_xy(
            list(map(lambda x: x["training_accuracy"], data))
        )
        accuracy =  "<h2>Accuracy</h2>" + accuracy_plot
        
    predictions = list(map(lambda x: get_prediction_format(x), data))[-5:]
    if len(loss_plot):
        loss_plot = ["<h2>Loss</h2>"] + loss_plot
    return "<br>".join([
        "<br>".join(loss_plot),
        accuracy,
        "<h1>Metrics</h1>",
        "<br>".join(predictions),
    ])

@app.route('/<project_name>/<run_id>', methods=['GET'])
def run(project_name, run_id):
    """
    Shows the project run data
    """
    if run_id == "latest":
        run_ids = SimpleDatabase.get_sorted_runs(project_name)
        if len(run_ids):
            run_id = run_ids[0]
    results = load_run_id(project_name, run_id)
    if results is None:
        return f"Unknown run id {run_id}"
    return results

def get_prediction_format(entry):
    value = [
        "<h3>{epoch}</h3>".format(epoch=entry["epoch"])
    ]
    if entry["prediction"] is None:
        value.append("*no metrics*")
    elif entry["prediction"]["prediction_type"] == "text":
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
