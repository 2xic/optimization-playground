import json
from dataset import Dataset
from utils import shuffle, features
import numpy as np

def get_dataset_features():
    for index in range(len(features)):
        include, exclude = None, None
        with open("config.json", "r") as file:
            json_data = json.loads(file.read())
            include, exclude = json_data["include"], json_data["exclude"]

        dataset = Dataset().load(
            exclude=exclude,
            include=include,
        )

        new_features = features[:index] + features[index + 1: ]
        x, y, x_test, y_test = dataset.get_x_y(
            features=new_features,
            split=0.8,
            adjust_n_samples=False
        )
        x, y = shuffle(x, y)
        x = np.asarray(x)
        x_test = np.asarray(x_test)
        x[:, index] = 0
        x_test[:, index] = 0
        yield x, y, x_test, y_test, features[index]
