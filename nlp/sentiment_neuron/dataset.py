import json

class Dataset:
    def __init__(self, dataset_entries) -> None:
        self.data = None
        with open("reviews_Video_Games.json", "r") as file:
            self.data = file.read().split("\n")
        self.max_data_entries = dataset_entries

    def get_text(self, max_size):
        sentences = []
        for i in self.data[:self.max_data_entries]:
            try:
                i = json.loads(i)
                sentences.append(i["reviewText"].lower()[:max_size])
            except Exception as e:
                print(i)
                print(e)
        return sentences

    def get_score_and_text(self, max_size):
        sentences = []
        scores = []
        for i in self.data[:self.max_data_entries]:
            try:
                i = json.loads(i)
                sentences.append(i["reviewText"].lower()[:max_size])
                scores.append(i["overall"] / 5)
            except Exception as e:
                print(i)
                print(e)
        return sentences, scores
