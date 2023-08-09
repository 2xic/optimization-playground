from flask import Flask, request
from flask import render_template
from model import get_model
from utils import build_suggestion
from api import Api
from move_songs import MoveSongs
import atexit
from utils import features

best_model, dataset = get_model()
combined, reorganize_playlist = build_suggestion(best_model, dataset)
moved = MoveSongs()
app = Flask(__name__)
api = Api(
    is_cache_only_mode=False
)


def save_move():
    moved.save()


atexit.register(save_move)


@app.route("/")
def index():
    suggestions = list(filter(
        lambda x: x["song"].id not in moved.list,
        combined
    ))[:400]
    return render_template('frontend.html', suggestions=suggestions)


@app.route("/song", methods=["GET"])
def api_get_song_features():
    song_id = request.args.get("song_id")
    return api.get_song_feature(song_id)


@app.route("/recommendations", methods=["GET"])
def api_get_song_playlist_recommendation():
    song_id = request.args.get("song_id")
    song_features = list(api.get_song_feature(song_id))[0]
    print(song_features)
    dataset_features = dataset.get_song_features(
        song_features,
        features,
    )
    playlist_id = dataset.class_id[best_model.predict([
        dataset_features
    ])[0]]
    return dataset.ids_to_name[playlist_id]


@app.route("/play", methods=["POST"])
def play():
    song_id = request.args.get("song_id")
    api.play_song(song_id)
    return "OK"


@app.route("/move_song", methods=["POST"])
def server_move_song():
    song_id = request.args.get("song_id")
    playlist_id = request.args.get("playlist_id")
    api.move_song(
        song_id,
        reorganize_playlist,
        playlist_id,
    )
    moved.move(song_id)

    return "OK"

if __name__ == "__main__":
    app.run()
