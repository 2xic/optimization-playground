import torch
from .models.EmbeddingsModel import EmbeddingsModel

class CbowModel:
    def __init__(self, vocab, device) -> None:
        self.vocab = vocab
        vocab_size = self.vocab.size

        self.model = EmbeddingsModel(vocab_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters()
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device

    def fit(self, X, y):
        output = self.model(X)
#        print(y)
        loss = self.loss(output, y.reshape((-1)).long())
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, text):
        indexes = self.vocab.get(text.lower().split(" "))
        encoded = torch.zeros((len(indexes), 4)).to(self.device)
        for index, i in enumerate(indexes):
            encoded[index][0] = i
        output = self.model(encoded.long())
        return output
