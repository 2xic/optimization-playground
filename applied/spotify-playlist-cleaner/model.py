from models.model_search import find_model
from models.feature_search import get_dataset_features
from utils import load_dataset, get_distribution_playlist_predictions

def get_model():
    x, y, x_test, y_test, dataset = load_dataset()

    best_model, best_accuracy = find_model(
        x, y, x_test, y_test
    )
    print(f"Best acc : {best_accuracy}")
    return best_model, dataset

def get_features():
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
    best_model, dataset = get_model()

    if give_distribution:
        print(get_distribution_playlist_predictions(
            best_model,
            dataset
        ))
    get_features()
    