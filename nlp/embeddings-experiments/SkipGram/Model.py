import torch
import torch.nn as nn

class Model(torch.nn.Module):
    def __init__(self, vocab_size) -> None:
        super().__init__()
        self.embedding_layer = nn.Embedding(vocab_size, 32)
        prev_layer =  32
        layers = [
            vocab_size * 2,
            vocab_size * 4,
            vocab_size * 2,
            vocab_size,
        ]
        torch_layers = []
        for i in layers:
            torch_layers.append(
                nn.Linear(
                    prev_layer, i
                )
            )
            torch_layers.append(
                nn.Sigmoid()
            )
            prev_layer = i

        self.model = nn.Sequential(
            *torch_layers,
        #    nn.Sigmoid(),
        #    nn.Softmax(dim=1)
        )
    
    def forward(self, X):
        emb = self.embedding_layer(X)
      #  print(emb.shape)
        return self.model(emb.reshape(X.shape[0], -1))
    
class SkipGramModel:
    def __init__(self, vocab, device) -> None:
        self.vocab = vocab
        vocab_size = self.vocab.size

        self.model = Model(vocab_size).to(device)
        print(self.model)
        self.optimizer = torch.optim.Adam(
            self.model.parameters()
        )
#        self.loss = torch.nn.NLLLoss()
        self.loss = torch.nn.CrossEntropyLoss()
        self.device = device

    def fit(self, X, y):
        output = self.model(X)

#        loss = ((output - y) ** 2).mean()
#        print(output)
        loss = self.loss(output, y)#.reshape(-1))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def predict(self, text):
        indexes = self.vocab.get(text.lower().split(" "))
        encoded = torch.zeros((len(indexes), 1)).to(self.device)
        for index, i in enumerate(indexes):
            encoded[index][0] = i
        output = self.model(encoded.long())
        print(output)
        return output
#        return output.mean(dim=-1)
