from flask import Flask, request, jsonify

def create_route(app, model):
    @app.route("/predict", methods=["POST"])
    def predict():
        data = request.json
        text = data["text"]
        try:
            embedding = model.transforms([text])
            embedding = embedding[0]
            return jsonify({
                "embedding": embedding.tolist(),
            })
        except Exception as e:
            return jsonify({
                "embedding": None,
            })
    return predict

def create_flask_app(model):
    app = Flask(__name__)
    create_route(
        app,
        model
    )
    return app
