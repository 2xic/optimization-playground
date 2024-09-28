from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestRegressor
from sklearn import svm
import matplotlib.pyplot as plt
import os
from dataset import get_dataset
from pipeline import Pipeline
from embeddings import TfIdfWrapper, OpenAiEmbeddingsWrapper, HuggingFaceWrapper, ClaudeWrapper
from optimization_playground_shared.classics.bm_25 import BM25
from torch_gpt_like_model import EmbeddingWrapper
from torch_contrastive_model import ContrastiveEmbeddingWrapper
from torch_gpt_like_model_bigger import EmbeddingWrapperBigger
from xgboost import XGBRegressor
from optimization_playground_shared.utils.ClassImbalanceSplitter import balance_classes
from big_embeddings.Wrapper import Wrapper

def evaluation():
    X, y = get_dataset()
    X, y = balance_classes(X, y)
    X = [str(i) for i in X]
    assert type(X[0]) == str, type(X[0])
    X_train_original, X_test_original, y_train_original, y_test_original = train_test_split(
        X, y, test_size=0.33, random_state=42
    )
    model_pipeline_configs = {
        "big_gpt": [
            Wrapper(),
        ],
        "BM25": [
            BM25(),
        ],
#        "torch_next_token_bigger": [
#            EmbeddingWrapperBigger(),
#            EmbeddingWrapperBigger().load(".model_state_gpt_bigger_old_good_one.pkt"),
#            EmbeddingWrapperBigger().load(".model_state_gpt_bigger_lr.pkt"),
#        ],
#        "torch_contrastive": [
#            ContrastiveEmbeddingWrapper(),
#        ],
#        "untrained reference": [
#            EmbeddingWrapper(trained=False),
#        ],
#        "torch_next_token": [
#            EmbeddingWrapper(),
#            EmbeddingWrapper().load(".model_state_gpt_lr.pkt"),
#        ],
        "mixedbread-ai":[
            HuggingFaceWrapper(
                "mixedbread-ai/mxbai-embed-large-v1"
            ),
        ],
        "intfloat":[
            HuggingFaceWrapper("intfloat/multilingual-e5-large"),
        ],
        "tf_idf": [
            TfIdfWrapper(max_features=75),
            TfIdfWrapper(max_features=100),
            TfIdfWrapper(max_features=150),
        ],
        "open_ai": [
            OpenAiEmbeddingsWrapper("text-embedding-ada-002"),
            OpenAiEmbeddingsWrapper("text-embedding-3-large"),
            OpenAiEmbeddingsWrapper("text-embedding-3-small"),
        ],
        "voyage": [
            ClaudeWrapper(),
        ],
    }
    results = {}
    for base_config_name in model_pipeline_configs:
        best_local_config_score = 0
        best_local_config_string = None
        for index, embedding in enumerate(model_pipeline_configs[base_config_name]):
            (X_train, X_test, y_train, y_test) = Pipeline(
                sample_size=1
            ).transform(
                X_train_original, X_test_original, y_train_original, y_test_original, embedding
            )
            # TODO: Add more fancy models also
            models = [
                RandomForestRegressor(max_depth=2, random_state=0),
                RandomForestRegressor(max_depth=8, random_state=0),
                RandomForestRegressor(max_depth=4, random_state=0),
                svm.SVR(),
                XGBRegressor(),
            ]
            for model in models:
                config_name =  f"{model.__class__.__name__} + {base_config_name}"
                print((config_name))
                model.fit(X_train, y_train)
                accuracy = accuracy_score(y_test, list(map(lambda x: min(max(round(x), 0), 1), model.predict(X_test))))
                print(f"\t{model.__class__.__name__} -> accuracy: {accuracy}")
                print("")
                new_score = accuracy * 100
                if new_score > best_local_config_score:
                    best_local_config_score = new_score
                    best_local_config_string = config_name
            # Clear the memory bag.
            model_pipeline_configs[base_config_name][index] = None
        # clear the model to free up memory.
        model_pipeline_configs[base_config_name] = []
        results[best_local_config_string] = best_local_config_score

    results = {
        key:value
        for key, value in sorted(results.items(), key=lambda x: x[1])
    }
    config_name = list(results.keys())
    values = list(results.values())
    
    plt.bar(config_name, values,width = 0.4)
    plt.xlabel("Config name")
    plt.ylabel("Accuracy %")
    plt.ylim([50, 100])
    plt.title("Results")
    path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "results.png"
    )
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path)

if __name__ == "__main__":
    evaluation()
