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
from .resource_tracker import ResourceTracker

root_data_directory = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".data"
)
global_metadata = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".data",
    "metadata.json"
)

resource_tracker = ResourceTracker()

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
        previous_data = {}
        if os.path.isfile(file_path):
            with open(file_path, "r") as file:
                previous_data = json.load(file)
        for key, value in dataclasses.asdict(metric).items():
            if value is not None:
                previous_data[key] = value
        with open(file_path, "w") as file:
            json.dump(previous_data, file)
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
        return list(map(lambda x: x[0], SimpleDatabase.get_sorted_runs_with_timestamp(project_name)))

    @staticmethod
    def get_sorted_runs_with_timestamp(project_name):
        project_metadata_path = SimpleDatabase.metadata_file(project_name)
        with open(project_metadata_path, "r") as file:
            data = json.load(file)
            ids_timestamp = []
            for (run_id, data) in sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True):
                ids_timestamp.append((run_id, data["last_commit"]))
            return ids_timestamp

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

def get_main_runs():
    if os.path.isfile(global_metadata):
        with open(global_metadata, "r") as file:
            data = json.load(file)
            return sorted(data.items(), key=lambda x: x[1]["last_commit"], reverse=True)
    return []
            

@app.route('/', methods=['GET'])
def index():
    """
    Shows the projects data
    """
    main_runs = get_main_runs()
    if len(main_runs) > 0:
        results = []
        for (name, _) in main_runs:
            print(name)
            results.append(
                f"<a href={name}>{name}</a>".format(name=name)
            )
        return "<br>".join(results)
    else:
        return "No projects found, have you executed any experiments?"

@app.route('/table', methods=['GET'])
def table():
    """
    Shows the projects data
    """
    table = """
        <table>
        <tr>
            <th>Name</th>
            <th>Accuracy</th>
            <th>Epochs</th>
        </tr>
    """
    rows = []
    for (project_name, _) in get_main_runs():
        runs = SimpleDatabase.get_sorted_runs_with_timestamp(project_name)
        if len(runs) == 0:
            continue
        for (i, timestamp) in runs:
            output = load_run_id_epoch_data(project_name, i)[-1]
            accuracy = output.get("training_accuracy", None)
            epochs = output["epoch"]
            if accuracy is None:
                continue
            real_run_id = f"<a href={project_name}/{i}>{project_name}</a>"
            rows.append(("\n".join([
                "<tr>",
                    f"<td>{real_run_id}</td>",
                    f"<td>{accuracy}</td>",
                    f"<td>{epochs}</td>",
                "</tr>",
            ]), timestamp))
    table += "\n".join(
        list(map((lambda x: x[0]), sorted(rows, key=lambda x: x[1], reverse=True))))
    table += "</table>"
    return table


@app.route('/<project_name>', methods=['GET'])
def runs(project_name):
    """
    Shows the project run data
    """
    if not os.path.isfile(SimpleDatabase.metadata_file(project_name)):
        return "Project name not found, have you executed the experiment yet?"

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

def load_run_id_epoch_data(project_name, run_id):
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
    return data

def load_run_id(project_name, run_id, n):
    data = load_run_id_epoch_data(project_name, run_id)
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
    if not all(list(map(lambda x: x.get("training_accuracy", None) is None, data))):
        accuracy_plot = plot_xy(
            list(map(lambda x: x["training_accuracy"], data))
        )
        accuracy =  "<h2>Accuracy</h2>" + accuracy_plot
    
    start_timestamp = (data[-1]["timestamp"] - data[0]["timestamp"]) / data[-1]["epoch"] if len(data) > 1 else None
    print("timestamp ", data[-1]["timestamp"])
    print("timestamp ", data[0]["timestamp"])
    predictions = list(map(lambda x: get_prediction_format(x), data))[-n:]
    # Reverse so you can see the latest results first 
    predictions = predictions[::-1]
    
    if len(loss_plot):
        loss_plot = ["<h2>Loss</h2>"] + loss_plot
    
    output = [
        "<h1>meta</h1>",
        f"epoch iterations {start_timestamp}" if start_timestamp is not None else "No iterations yet",
        "<br>".join(loss_plot),
        accuracy,
        "<h1>Metrics</h1>",
        "<br>".join(predictions),
    ]
    return "<br>".join(output)

@app.route('/<project_name>/<run_id>', methods=['GET'])
def run(project_name, run_id):
    """
    Shows the project run data
    """
    page_size = request.args.get("n", 5)
    if run_id == "latest":
        run_ids = SimpleDatabase.get_sorted_runs(project_name)
        if len(run_ids):
            run_id = run_ids[0]
    results = load_run_id(project_name, run_id, page_size)
    if results is None:
        return f"Unknown run id {run_id}"
    return results

@app.route('/resource_usage', methods=['GET'])
def resource_usage():
    # resource_tracker
    plot = Plot()
    path = plot.plot_figures(
        figures=[
            Figure(
                plots={
                    "percentage":resource_tracker.cpu,
                },
                title="Cpu usage",
                y_axes_text="%",
            ),
            Figure(
                plots={
                    "percentage":resource_tracker.ram,
                },
                title="ram usage",
                y_axes_text="%",
            ),
            Figure(
                plots={
                    "percentage":resource_tracker.gpu,
                },
                title="gpu usage",
                y_axes_text="%",
            ),
            Figure(
                plots=resource_tracker.gpus,
                title="gpus usage",
                y_axes_text="%",
            ),
        ],
        name='.training.png',
        show_legend=False,
    )
    with open(path, "rb") as file:
        encoded_data = base64.b64encode(file.read()).decode("ascii")
        return f"<img src='data:image/png;base64,{encoded_data}'/>"

@app.route('/resource_usage', methods=['POST'])
def update_resource_usage():
    global resource_tracker
    content = request.json
    for key, value in content.items():
        resource_tracker.add_usage(key, value)
    return jsonify({
        "success": True,
    })

def get_prediction_format(entry):
    value = [
        "<h3>{epoch}</h3>".format(epoch=entry["epoch"])
    ]
    if entry.get("prediction", None) is None:
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

