import torch.nn as nn
import torch
from T_dmca import AttentionLayer
from vocab import Vocab

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

   #     vocab_size = 4
        position_size = 4
        self.vocab_size = vocab_size
        embedding_dim = 8

        self.word_embedding = nn.Embedding(
            vocab_size, embedding_dim=embedding_dim)
        self.position_embedding = nn.Embedding(
            position_size, embedding_dim=embedding_dim)

        self.blocks = torch.nn.Sequential(*[
            AttentionLayer(8),
        ])
        self.lm_head = nn.Linear(embedding_dim * 4, vocab_size, bias=False)

    def forward(self, text):
        #print("shape ", text.shape)
        position = torch.arange(0, text.shape[1], dtype=torch.long).unsqueeze(0) # shape (1, t)
        h_0 = self.word_embedding(text) + self.position_embedding(position)
        h_1 = self.blocks(h_0)

     #   print(h_1.shape)

        return torch.softmax(self.lm_head(h_1.reshape((1, 4 * 8))), dim=1)


if __name__ == "__main__":
    vocab = Vocab()
    X_0 = vocab.encode("hello world, have a nice day ! wowow ")
    X_1 = vocab.encode("good night, and hello world, have a nice night ! wowow ")
    X_2 = vocab.encode("hello my friend, how are you doing ? ")
    X_3 = vocab.encode("what is a friend ? it is a good human ")
    X_4 = vocab.encode("What is I ? I am, machine")

    model = Model(vocab.lock())
    optimizer = torch.optim.Adam(model.parameters())

    for _ in range(1_000):
        loss_item = 0
        optimizer.zero_grad()
        for X in [X_0, X_1, X_2, X_3, X_4]:
            for i in range(X.shape[1] - 5):
                out = model(X[:, i:i+4])
                next_token = torch.argmax(out, dim=1)
                expected_next_token = X[0][i + 4 + 1].reshape((1))

                loss = torch.nn.CrossEntropyLoss()(
                    out,
                    expected_next_token
                )
                loss.backward()
                loss_item += loss.item()
        optimizer.step()
        print(loss_item)

    for i in [X_0, X_2, X_3]:
        index = 0
        out = model(i[:, index:index+4])
        next_token = torch.argmax(out, dim=1)
        print([vocab.decode(i) for i in i[0, index:index+4].tolist()])
        print(next_token, vocab.decode(next_token.item()))