import os
from flask import Flask, render_template
from flask import Flask
import json
import argparse


app = Flask(__name__)

model_rows = {}

@app.route('/data')
def data():
    return json.dumps(model_rows)

@app.route('/code/<page_id>')
def code(page_id):
    with open(f"cached/{page_id}") as file:
        return file.read()

@app.route('/')
def index():
    return render_template('index.html')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog='visualization')
    parser.add_argument('-f', '--file',  nargs='+', required=True)

    args = parser.parse_args()
    
    dump_file = args.file
    for model_dump_file in dump_file:    
        print(model_dump_file)    
        if os.path.isfile(model_dump_file):
            with open(model_dump_file, "r") as file:
                rows = json.load(file)
                model_rows[rows["name"]] = rows["rows"]
        else:
            print("Run the train file first")
            exit(0)
    print(model_rows.keys())
    app.run(
        host='0.0.0.0'
    )
