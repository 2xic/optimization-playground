from models.model_search import find_model
from models.feature_search import get_dataset_features
from utils import load_dataset, get_distribution_playlist_predictions
import pickle
import os
from optimization_playground_shared.clustering.ElbowKmeans import ElbowKmeans

def train_cluster_model():
    X, y, _, _, _ = load_dataset()
    model = ElbowKmeans()
    return model.fit(X, y)

def get_or_train_model():
    cache_path = ".cached_model"
    x, y, x_test, y_test, dataset = load_dataset()

    if os.path.isfile(cache_path):
        with open(cache_path, "rb") as file:
            model = pickle.load(file)
            return model, dataset

    best_model, best_accuracy = find_model(
        x, y, x_test, y_test
    )
    print(f"Best acc : {best_accuracy}")
    with open(cache_path, "wb") as file:
        pickle.dump(best_model, file)
    return best_model, dataset

def evaluate_features():
    for (x, y, x_test, y_test, feature) in get_dataset_features():
        _, best_accuracy = find_model(
            x, 
            y, 
            x_test, 
            y_test,
            log=False
        )
        print(f"Training without feature: {feature} ")
        print(f"Best acc : {best_accuracy}")

if __name__ == "__main__":
    give_distribution = False
    should_evaluate_features = False
    best_model, dataset = get_or_train_model()

    if give_distribution:
        print(get_distribution_playlist_predictions(
            best_model,
            dataset
        ))
    if should_evaluate_features:
        evaluate_features()
    