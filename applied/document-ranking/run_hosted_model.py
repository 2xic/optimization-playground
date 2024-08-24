from flask import Flask, request, jsonify
from torch_gpt_like_model_bigger import EmbeddingWrapperBigger


models = {
    "gpt_like_model": EmbeddingWrapperBigger().load(".model_state_gpt_bigger_lr.pkt"),
}

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    text = data["text"]
    model = models[data["model"]]
    embedding = model.transforms([text])
    embedding = embedding[0]
    return jsonify({
        "embedding": embedding.tolist(),
    })

if __name__ == "__main__":
    app.run(
        port=1245
    )
