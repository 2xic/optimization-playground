import random
import json
from dataset import Dataset
from collections import defaultdict, namedtuple

features = [
    "mode", 
    "danceability",
    "acousticness",
    "liveness",
    "valence",
    "tempo",
    "speechiness",
    "energy",
]

def shuffle(x, y):
    temp = list(zip(x, y))
    random.shuffle(temp)
    res1, res2 = zip(*temp)
    res1, res2 = list(res1), list(res2)
    return res1, res2

def load_dataset():
    include, exclude = None, None
    with open("config.json", "r") as file:
        json_data = json.loads(file.read())
        include, exclude = json_data["include"], json_data["exclude"]

    dataset = Dataset().load(
        exclude=exclude,
        include=include,
    )
    x, y, x_test, y_test = dataset.get_x_y(
        features=features,
        split=0.8,
        adjust_n_samples=False
    )
    assert len(x) == len(y) and len(x) > 0
    x, y = shuffle(x, y)
    return x, y, x_test, y_test, dataset

def get_reorganize_playlist():
    reorganize_playlist = None
    with open("config.json", "r") as file:
        json_data = json.loads(file.read())
        reorganize_playlist = json_data["reorganize_playlist"]
    return reorganize_playlist

def reorganize_playlist(model, dataset: Dataset):
    reorganize_playlist = get_reorganize_playlist()
    x_pred, songs = Dataset().load_playlist(reorganize_playlist).get_song_prediction(
        reorganize_playlist,
        features=features
    )
    for x, song in zip(x_pred, songs):
        song_name = song.name
        playlist = dataset.ids_to_name[dataset.class_id[model.predict([x])[0]]]
        print(song_name)
        print(f"\t{playlist}")

def get_distribution_playlist_predictions(model, dataset):
    reorganize_playlist = get_reorganize_playlist()
    x_pred, _ = Dataset().load_playlist(reorganize_playlist).get_song_prediction(
        reorganize_playlist,
        features=features
    )
    distribution = defaultdict(int)
    for x in x_pred:
        playlist = dataset.ids_to_name[dataset.class_id[model.predict([x])[0]]]
        distribution[playlist] += 1
    return distribution

def build_suggestion(model, dataset: Dataset):
    print("Building suggestions ...")
    predictions = namedtuple('prediction', ['playlist_id', 'playlist_name', 'predicted'])
    reorganize_playlist = get_reorganize_playlist()
    print("Building dataset ...")
    x_pred, songs = Dataset().load_playlist(reorganize_playlist).get_song_prediction(
        reorganize_playlist,
        features=features
    )
    combined = []
    print(dataset.ids_to_name)
    print("Predicting ...")
    for x, song in zip(x_pred, songs):
        playlist_id = dataset.class_id[model.predict([x])[0]]
       # playlist_name = dataset.ids_to_name[playlist_id]
        combined.append({
            "song": song,
            "prediction": [
                predictions(playlist_id=key, playlist_name=value, predicted=(
                    key == playlist_id
                ))
                for key, value in dataset.ids_to_name.items()
            ], 
        })
    return combined, reorganize_playlist
