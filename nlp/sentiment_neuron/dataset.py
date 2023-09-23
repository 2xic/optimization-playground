import json

class Dataset:
    def __init__(self) -> None:
        self.data = None
        with open("reviews_Video_Games.json", "r") as file:
            self.data = file.read().split("\n")
        self.max_size = 1_000

    def get_text(self):
        sentences = []
        for i in self.data[:self.max_size]:
            try:
                i = json.loads(i)
                sentences.append(i["reviewText"])
            except Exception as e:
                print(i)
                print(e)
        return sentences
