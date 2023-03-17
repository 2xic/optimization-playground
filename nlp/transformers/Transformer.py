import torch.nn as nn
from EncoderBlock import EncoderBlock
from utils import Tokenizer
import torch
from PositionalEncoding import PositionalEncoding

class Transformer(nn.Module):
    def __init__(self, tokens) -> None:
        super().__init__()

        self.embedding_dims = 16
        self.d_model = self.embedding_dims

        self.embeddings = nn.Embedding(
            tokens,
            self.embedding_dims
        )
        num_layers = 4
        self.encoding_blocks = nn.ModuleList([
            EncoderBlock(
                self.embedding_dims
            ) for _ in range(num_layers)
        ])

        self.output = nn.Sequential(*[
            nn.Linear(
                self.embedding_dims, tokens,
            ),
            nn.Sigmoid(),
            nn.Linear(
                tokens, tokens,
            ),
            nn.Sigmoid()
        ])

    def forward(self, x):
        x = self.embeddings(x)
        y = PositionalEncoding().encode_tensor(x.size(0), d_model=(self.d_model * x.size(0))).reshape((
            x.shape
        ))
        x += y
#        print(y.shape)
#        print(x.shape)
        for encoder_block in self.encoding_blocks:
            x = encoder_block(x)
        x = self.output(x)        
        return x

if __name__ == "__main__":
    documents = [
        "test sentence hello sir",
        "this is also a sentence",
        "what is a hello",
        "what is a sentence"
    ]
    targets = [
        1,
        1,
        0,
        0,
    ]
    tokenizer = Tokenizer().encode_documents(documents)
    model = Transformer(
        len(tokenizer.word_idx)
    )
    optimizer = torch.optim.Adam(
        model.parameters()
    )
    X = torch.concat(
        [tokenizer.encode_document_tensor(x, sequence_length=4) for x in documents]
    )
    y_truth_labels = torch.concat(
        [tokenizer.encode_document_tensor(x, sequence_length=4).flip(dims=(1, ))
        for x in documents]
    )
    y_truth = y_truth_labels.view(-1)
    for i in range(1_000):
        y_pred = model(X)
        prediction = y_pred.view(-1,y_pred.size(-1))
        
        optimizer.zero_grad()
        error = nn.CrossEntropyLoss()(
            prediction.float(),
            y_truth
        )
        error.backward()
        optimizer.step()
        
        acc = ((y_pred.argmax(dim=-1) == y_truth_labels).float().sum(dim=1)).mean()
        if i % 100 == 0:
            print(acc)
    model.eval()
    print(
        documents
    )        
    print(
        [
            tokenizer.decode(i) for i in model(X).argmax(dim=-1)
        ]
    )
