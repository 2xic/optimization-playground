import os
import json

def get_plot_path(file):
    name = os.path.basename(os.path.abspath(file)).replace(".py", ".png")
    path = os.path.abspath(os.path.dirname(file))

    dirname = os.path.join(
        path,
        "plot",
    )
    os.makedirs(dirname, exist_ok=True)

    return os.path.join(
        dirname,
        name
    )

def dump_agent_data(file, data):
    name = os.path.basename(os.path.abspath(file)).replace(".py", ".json")
    path = os.path.abspath(os.path.dirname(file))

    dirname = os.path.join(
        path,
        "data",
    )
    os.makedirs(dirname, exist_ok=True)

    path = os.path.join(
        dirname,
        name
    )
    with open(path, "w") as file:
        json.dump(data, file)
