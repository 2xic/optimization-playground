from dotenv import load_dotenv
import os
import json
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt

load_dotenv()

def get_path_files():
    path = {}
    path_colors = {}

    colors = ["red", "blue", "green"]


    for index, i in enumerate(os.environ["playlists"].split(",")+ ["shared="]):
        name,file = i.split("=")
        path[name] = file
        path_colors[name] = colors[index]

    return path, path_colors

def get_tracks():
    path, path_colors = get_path_files()
    playlist_name_tracks = {}
    track_ids = []
    prediction = []

    for name, file in path.items():
        if name == "shared":
            items = list(playlist_name_tracks.values())
            first_entry = set(items[0])
            for i in items[1:]:
                first_entry = set(i) & first_entry
            playlist_name_tracks[name] = first_entry
        else:
            with open(file, "r") as file:
                data = json.load(file)
                playlist_name_tracks[name] = data
                for track_id in data:
                    if track_id not in track_ids:
                        file = f".lyrics/{track_id}.json"
                        if os.path.isfile(file):
                            embeddings = None
                            with open(file, "r") as file:
                                content = json.load(file)
                                embeddings = content.get("embeddings", None)
                            if embeddings is None:
                                continue
                            track_ids.append(track_id)
                            prediction.append(embeddings)

    positions_tracks = {}
#    tsne = TSNE(random_state=1, n_iter=15000, metric="cosine", init='random', learning_rate='auto')
    tsne = TSNE(random_state=1, metric="cosine", init='random', learning_rate='auto')
    embedding_position = tsne.fit_transform(np.asarray(prediction))

    for track_id, position in zip(track_ids, embedding_position):
        positions_tracks[track_id] = position

    return playlist_name_tracks, positions_tracks, path_colors

def plot_embeddings():
    playlist_name_tracks, positions_tracks, path_colors = get_tracks()
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for playlist, tracks in playlist_name_tracks.items():
        x = []
        y = []
        for i in tracks:
            x.append(positions_tracks[i][0])    
            y.append(positions_tracks[i][1])    
        ax.scatter(x, y, color=path_colors[playlist], label=playlist)

    plt.legend()
    plt.savefig("playlists_embeddings_visualization.png")

if __name__ == "__main__":
    plot_embeddings()

