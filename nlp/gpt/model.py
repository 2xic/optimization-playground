import torch.nn as nn
import torch
from T_dmca import AttentionLayer
from vocab import Vocab


SEQ_SIZE = 4

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

   #     vocab_size = 4
        position_size = 64
        self.vocab_size = vocab_size
        self.embedding_dim = 32

        self.word_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim=self.embedding_dim,
         #   padding_idx=-1
        )
        self.position_embedding = nn.Embedding(
            position_size, 
            embedding_dim=self.embedding_dim,
        #    padding_idx=-1
        )

        self.blocks = torch.nn.Sequential(*[
            AttentionLayer(32),
            #AttentionLayer(32),
        ])
        self.lm_head = nn.Linear(self.embedding_dim * 4, vocab_size, bias=False)

    def forward(self, text, position):
        #print("shape ", text.shape)
        h_0 = self.word_embedding(text) + self.position_embedding(position)
        h_1 = self.blocks(h_0)

     #   print(h_1.shape)
        
        return torch.softmax(self.lm_head(h_1.reshape((text.shape[0], 4 * self.embedding_dim))), dim=1)


if __name__ == "__main__":
    vocab = Vocab()
    X = vocab.encode_file("large_tet_file")
#    print(X.shape)

    model = Model(vocab.lock())
    optimizer = torch.optim.Adam(model.parameters())
    count_parameters(model)
#    exit(0)

    for _ in range(1_000):
        loss_item = 0
        optimizer.zero_grad()
        #for X in [X_0, X_1, X_2, X_3, X_4]:
        for i in range(X.shape[1] - (SEQ_SIZE + 1)):
            position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) + i # shape (1, t) 
            out = model(X[:, i:i+SEQ_SIZE], position)
            next_token = torch.argmax(out, dim=1)
            expected_next_token = X[:, i + SEQ_SIZE + 1].reshape((X.shape[0]))

            loss = torch.nn.CrossEntropyLoss(
                ignore_index=vocab.padding_idx
            )(
                out,
                expected_next_token,
            )
            loss.backward()
            loss_item += loss.item()
        optimizer.step()
        print(loss_item)

    for i in [X[0], X[1], X[2]]:
        position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) #+ i # shape (1, t) 
        i = i.reshape((1, ) + i.shape)
        index = 0
        out = model(i[:, index:index+4], position)
        next_token = torch.argmax(out, dim=1)
        print([vocab.decode(i) for i in i[0, index:index+4].tolist()])
        print(next_token, vocab.decode(next_token.item()))
