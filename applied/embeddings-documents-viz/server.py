import os
from flask import Flask, render_template
from flask import Flask
import json
import argparse

app = Flask(__name__)

data = {}

@app.route('/data')
def data():
    return json.dumps(data)

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='visualization')
    parser.add_argument('-f', '--file',  required=True)
    args = parser.parse_args()
    assert os.path.isfile(args.file), "File is not found, run the build data file"

    with open(args.file, "r") as file:
        data = json.load(file)

    app.run(
        host='0.0.0.0'
    )
