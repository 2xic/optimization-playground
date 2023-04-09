import os
import json

class MoveSongs:
    def __init__(self):
        self.list = self.load()

    def load(self):
        path = self.get_path()
        if os.path.isfile(path):
            with open(path, "r") as file:
                return json.loads(file.read())
        else:
            return {}
            
    def move(self, song_id):
        self.list[song_id] = True

    def save(self):
        path = self.get_path()
        with open(path, "w") as file:
            file.write(json.dumps(self.list))

    def get_path(self):
        path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "moved.json"
        )
        return path
