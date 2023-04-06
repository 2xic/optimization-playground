from flask import Flask, request
from flask import render_template
from model import get_model
from utils import build_suggestion
from api import play_song

best_model, dataset = get_model()
combined =  build_suggestion(best_model, dataset)

app = Flask(__name__)

@app.route("/")
def hello_world():
    return render_template('frontend.html', suggestions=combined)

@app.route("/play", methods=["POST"])
def play():
    song_id = request.args.get("song_id")
    print(song_id)
    return "OK"


if __name__ == "__main__":
    app.run()
