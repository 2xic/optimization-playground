from flask import Flask, request, jsonify
from .model_1 import EmbeddingModelOne
from .model_2 import EmbeddingModelTwo
from .model_3 import EmbeddingModelThree
from .model_4 import EmbeddingModelFour

all_models = [
    EmbeddingModelTwo("SimpleContrastiveLoss").load(),
    EmbeddingModelOne().load(),
    EmbeddingModelThree().load(),
    EmbeddingModelFour().load()
]

mapping = {}
for i in all_models:
    mapping[i.__class__.__name__] = i

    print(i.transforms(["hello, this is some text"]))
    print(i.transforms(["computer science"]))

app = Flask(__name__)

@app.route("/models")
def models():
    return jsonify({
        "models": list(mapping.keys())
    })

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]
    model = data["model"]
    model = mapping[model]
    embedding = model.transforms([text])
    embedding = embedding[0]
    return jsonify({
        "embedding": embedding.tolist(),
    })

if __name__ == "__main__":
    app.run(
        port=8081,
        host="0.0.0.0"
    )
