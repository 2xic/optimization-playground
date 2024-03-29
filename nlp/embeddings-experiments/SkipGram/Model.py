import torch
from .models.EmbeddingsModel import EmbeddingsModel
from .models.LinearModel import LinearModel

class SkipGramModel:
    def __init__(self, vocab, device) -> None:
        self.vocab = vocab
        vocab_size = self.vocab.size

        self.model = LinearModel(vocab_size).to(device)
        self.optimizer = torch.optim.Adam(
            self.model.parameters()
        )
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device

    def fit(self, X, y):
        output = self.model(X)

        loss = self.loss(output, y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, text):
        indexes = self.vocab.get(text.lower().split(" "))
        encoded = torch.zeros((len(indexes), 1)).to(self.device)
        for index, i in enumerate(indexes):
            encoded[index][0] = i
       # print(encoded)
        output = self.model(encoded.long())
      #  print(output)
        return output
