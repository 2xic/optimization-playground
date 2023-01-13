import torch.nn as nn
import torch
from T_dmca import AttentionLayer
from vocab import Vocab
from train_loop import train_loop
from optimization_utils.logging.EpochRuns import EpochRuns

def count_parameters(model):
    total_params = 0
    for _, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        total_params+=param
    print(f"Total Trainable Params: {total_params}")

class Model(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()

        self.SEQ_SIZE = 4
        self.position_size = 64
        self.vocab_size = vocab_size
        self.embedding_dim = 32

        self.word_embedding = nn.Embedding(
            vocab_size, 
            embedding_dim=self.embedding_dim,
         #   padding_idx=-1
        )
        self.position_embedding = nn.Embedding(
            self.position_size, 
            embedding_dim=self.embedding_dim,
        #    padding_idx=-1
        )

        self.blocks = torch.nn.Sequential(*[
            AttentionLayer(32),
            #AttentionLayer(32),
        ])
        self.lm_head = nn.Linear(self.embedding_dim * self.SEQ_SIZE, vocab_size, bias=False)
        self.log_softmax = nn.LogSoftmax(dim=1)
        
        self.epoch_information = EpochRuns(
            "gpt-like"
        )

    def forward(self, text, position):
        h_0 = self.word_embedding(text) + self.position_embedding(position)
        h_1 = self.blocks(h_0)

        return self.log_softmax(self.lm_head(h_1.reshape((text.shape[0], self.SEQ_SIZE * self.embedding_dim))))


if __name__ == "__main__":
    vocab = Vocab()
    X = vocab.encode_file("large_tet_file")
    model = Model(vocab.lock())
    optimizer = torch.optim.Adam(model.parameters())
    count_parameters(model)

    train_loop(
        model=model,
        vocab=vocab,
        X=X,
        optimizer=optimizer,
        epochs=1_000
    )
