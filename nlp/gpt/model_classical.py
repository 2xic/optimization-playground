import torch.nn as nn
import torch
from vocab import Vocab
from ClassicalAttention import ClassicalAttentionLayer

SEQ_SIZE = 8

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

        USE_ATTENTION = False

        self.ATTENTION_SHAPE = 1024
        self.LINEAR_SHAPE = self.embedding_dim * SEQ_SIZE

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

        self.block_converter = nn.Linear(self.LINEAR_SHAPE, self.ATTENTION_SHAPE)

        self.blocks = torch.nn.Sequential(*(
                [
                    nn.Sigmoid(),
                    ClassicalAttentionLayer(self.ATTENTION_SHAPE, self.ATTENTION_SHAPE),
                ]             
                if USE_ATTENTION 
                else [
                    nn.Sigmoid()
                ]
            )
        )
        self.lm_head = nn.Linear(self.ATTENTION_SHAPE, vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, text, position):
        #print("shape ", text.shape)
        h_0 = self.word_embedding(text) + self.position_embedding(position)
        h_0 = h_0.reshape((h_0.shape[0], self.embedding_dim * SEQ_SIZE))
        h_0 = torch.sigmoid(self.block_converter(h_0))

        h_1 = self.blocks(h_0)
        h_1 = torch.sigmoid(h_1)
        
        return self.log_softmax(self.lm_head(h_1.reshape((text.shape[0], self.ATTENTION_SHAPE))))


if __name__ == "__main__":
    vocab = Vocab()
    X = vocab.encode_file("large_tet_file")
#    print(X.shape)

    model = Model(vocab.lock())
    optimizer = torch.optim.Adam(model.parameters())
    count_parameters(model)
#    exit(0)

    loss_func = nn.NLLLoss(
        ignore_index=vocab.padding_idx
    )

  #  print(X.shape)
 #   print(vocab.lock())
#    exit(0)
    # torch.nn.CrossEntropyLoss
        
    for epoch in range(1_000):
        loss_item = 0

        #for X in [X_0, X_1, X_2, X_3, X_4]:
        for i in range(X.shape[1] - (SEQ_SIZE + 1)):
            optimizer.zero_grad()
            position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) + i
            out = model(X[:, i:i+SEQ_SIZE], position)
            next_token = torch.argmax(out, dim=1)
            expected_next_token = X[:, i + SEQ_SIZE].reshape((X.shape[0]))

            loss = loss_func(
                out,
                expected_next_token,
            )
            loss.backward()
            loss_item += loss.item()
            optimizer.step()
        print(loss_item, epoch)
      
    for i in [X[0], X[1], X[2]]:
        i = i.reshape((1, ) + i.shape)
        for index in range(0, 5):
            position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) + index
            out = model(i[:, index:index+SEQ_SIZE], position)
            next_token = torch.argmax(out, dim=1)
            if (0 == index):
                print([vocab.decode(i) for i in i[0, index:index+SEQ_SIZE].tolist()])
            print(f"{vocab.decode(next_token.item())} ({next_token.item()})")
        print("")

