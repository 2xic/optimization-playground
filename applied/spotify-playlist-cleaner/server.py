from flask import Flask, request
from flask import render_template
from model import get_model
from utils import build_suggestion
from api import play_song, move_song
from move_songs import MoveSongs
import atexit

best_model, dataset = get_model()
combined, reorganize_playlist =  build_suggestion(best_model, dataset)
moved = MoveSongs()
app = Flask(__name__)

def save_move():
    moved.save()

atexit.register(save_move)

@app.route("/")
def hello_world():
    suggestions = list(filter(
        lambda x: x["song"].id not in moved.list,
        combined
    ))[:400]
    return render_template('frontend.html', suggestions=suggestions)

@app.route("/play", methods=["POST"])
def play():
    song_id = request.args.get("song_id")
    play_song(song_id)
    return "OK"

@app.route("/move_song", methods=["POST"])
def server_move_song():
    song_id = request.args.get("song_id")
    playlist_id = request.args.get("playlist_id")
    move_song(
        song_id,
        reorganize_playlist,
        playlist_id,
    )
    moved.move(song_id)

    return "OK"

if __name__ == "__main__":
    app.run()
