from dataset import Dataset
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
from utils import shuffle
from model_search import find_model
import json
from utils import load_dataset, get_distribution_playlist_predictions

def get_model():
    x, y, x_test, y_test, dataset = load_dataset()

    best_model = find_model(
        x, y, x_test, y_test
    )
    return best_model, dataset

if __name__ == "__main__":
    best_model, dataset = get_model()
    
    print(get_distribution_playlist_predictions(
        best_model,
        dataset
    ))
