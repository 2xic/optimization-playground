from flask import Flask, request, jsonify

class ModelHost:
    def __init__(self):
        self.app = Flask(__name__)
        self.models = {}
        # Define routes
        self.app.add_url_rule('/predict', 'predict', self.predict, methods=["POST"])

    def add_model(self, name, instance):
        self.models[name] = instance
        return self

    def predict(self):
        data = request.json
        text = data["text"]
        model = self.models.get(data["model"])
        if model is None:
            return jsonify({"error": "Model not found"}), 404
        
        embedding = model.transforms([text])
        embedding = embedding[0]
        return jsonify({
            "embedding": embedding.tolist(),
        })

    def run(self):
        self.app.run(host="0.0.0.0", port=8081)


if __name__ == "__main__":
    app = ModelHost()
    app.run()
