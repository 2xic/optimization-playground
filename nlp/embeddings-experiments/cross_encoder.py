"""
https://arxiv.org/pdf/1908.10084.pdf

https://www.sbert.net/examples/applications/cross-encoder/README.html

Seems like the big difference is to not output the embedding and instead have NN 
to find the difference.
"""

import torch
import torch.nn as nn
from train_cbow import train_cbow, raw_text_dataset
from shared.get_sentence_sim import get_sentence_sim


class CrossEncoderModel(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.sequence = nn.Sequential(*[
            nn.Linear(self.base_model.output_size, 1),
            nn.Tanh()
        ])
        self.optimizer = torch.optim.Adam(self.sequence.parameters())

    def fit(self, X):
        # shuffle input tesnors
        shuffled_X = X.clone().detach()
        random_x = torch.randperm(X.shape[0])
        shuffled_X = shuffled_X[random_x[:, None], :]

        emb_X = self.get_base_embedding(X)
        emb_Y = self.get_base_embedding(shuffled_X)
        cos = nn.CosineSimilarity(dim=1, eps=1e-6)(emb_X, emb_Y)
        model = self.compare(X, shuffled_X).reshape(cos.shape)

        loss = nn.MSELoss()(model, cos)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss
    
    def predict(self, X, y):
        with torch.no_grad():
            emb_x = self.base_model.predict(X)
            emb_y = self.base_model.predict(y)
            return self._forward(
                emb_x.sum(dim=0).reshape(1, -1) / emb_x.shape[0],
                emb_y.sum(dim=0).reshape(1, -1) / emb_y.shape[0]
            )

    def compare(self, X, y):
        return self._forward(
            self.get_base_embedding(X),
            self.get_base_embedding(y)
        )
        
    def _forward(self, emb_x, emb_y):
        base_embedding = emb_x - emb_y
        output = self.sequence(base_embedding)
        return output

    def get_base_embedding(self, X):
        with torch.no_grad():
            return self.base_model.model(X)


def compare(cross_encoder, base_model, x, y):
    print(f"{x} / {y} > enc: {cross_encoder.predict(x, y).item()}")
    print(f"{get_sentence_sim(base_model, x, y)}")
    print("")

if __name__ == "__main__":
    epochs = 1_000
    model, dataloader = train_cbow(epochs)
    cross_encoder = CrossEncoderModel(model)
    
    for  epoch in range(epochs // 10):
        loss = 0
        for (X, _) in dataloader:
            loss += cross_encoder.fit(X)
        print(f"{epoch}: {loss.item()}")
    
    compare(
            cross_encoder,
            model,
            raw_text_dataset[0],
            raw_text_dataset[1]
    )
    compare(
            cross_encoder,
            model,
            raw_text_dataset[-2],
            raw_text_dataset[-1]
    )
    compare(
            cross_encoder,
            model,
            "bitcoin",
            "taco"
    )
    compare(
            cross_encoder,
            model,
            "taco for dinner",
            "taco is food"
    )    
    compare(
            cross_encoder,
            model,
            "smart contracts",
            "ethereum"
    )
    compare(
            cross_encoder,
            model,
            "ethereum has smart contracts",
            "taco is food"
    )
    