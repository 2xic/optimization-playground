from dataset import Dataset
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils import shuffle
from model_search import find_model
import json

include, exclude = None, None
reorganize_playlist = "reorganize_playlist"
with open("config.json", "r") as file:
    json_data = json.loads(file.read())
    include, exclude = json_data["include"], json_data["exclude"]
    reorganize_playlist = json_data["reorganize_playlist"]

dataset = Dataset().load(
    exclude=exclude,
    include=include,
)
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
x, y, x_test, y_test = dataset.get_x_y(
    features=features,
    split=0.8,
    adjust_n_samples=False
)
x, y = shuffle(x, y)

find_model(
    x, y, x_test, y_test
)

exit(0)


x_pred, songs = Dataset().load_playlist(reorganize_playlist).get_song_prediction(
    reorganize_playlist,
    features=features
)

for x, song in zip(x_pred, songs):
 #   print(clf.predict([x]))
 #   print(dataset.id_class)
    playlist = dataset.ids_to_name[dataset.class_id[clf.predict([x])[0]]]
    #print(f"'{song}' \tshould go into\t '{playlist}'")
    print(song)
    print(f"\t{playlist}")

