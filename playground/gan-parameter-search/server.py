"""
Simple server for visualizing of the gan parameters
"""
from flask import Flask, request
from flask import render_template
import json
from optimization_playground_shared.parameters_search.ParameterSearchWithFeedback import ParameterSearchState
import os
import glob


app = Flask(__name__)

def _get_parameter_epochs():
    a = {}
    for i in glob.glob(".parameter_search/**/*.json"):
        dirname = os.path.dirname(i).split("/")[-1]
        if str(dirname).isnumeric():
            a[dirname] = 1
    return list(range(len(a)))

@app.route("/")
def hello_world():
    with open(".parameter_search/metadata.json", "r") as file:
        data = json.loads(file.read())
        # todo find this dynamically
        data["epoch"] = _get_parameter_epochs()
        return render_template('index.html', metadata=json.dumps(data))

@app.route("/", methods=["POST"])
def get_image_state():
    print(request.get_json())
    parameters = request.get_json()
    epoch = parameters["epoch"]
    del parameters["epoch"]
    path = (ParameterSearchState.get_file_name(epoch, parameters))

    if os.path.isfile(path):
        with open(path, "r") as file:
            data = json.load(file)
            return data["generated_image"]["value"]
    return ""


app.jinja_env.auto_reload = True
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.run(debug=True)
