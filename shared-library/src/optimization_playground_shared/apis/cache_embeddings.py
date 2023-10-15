import os
import hashlib
import json

class CacheEmbeddings:
    def save(self, payload, results):
        model = payload["model"]
        text = payload["input"]

        file_path = self._get_path(
            model=model,
            text=text
        )
        with open(file_path, "w") as file:
            file.write(json.dumps(results))

    def load(self, model, input):
        file_path = self._get_path(
            model=model,
            text=input
        )
        if not os.path.isfile(file_path):
            return None
        with open(file_path, "r") as file:
            return json.loads(file.read())

    def _get_path(self, model, text):
        text_hash = hashlib.sha256(text.encode()).hexdigest()[:8]
        dir_path = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)
            ),
            ".cache",
            model
        )
        os.makedirs(dir_path, exist_ok=True)
        file_path = os.path.join(
            dir_path,
            text_hash
        )
        return file_path
