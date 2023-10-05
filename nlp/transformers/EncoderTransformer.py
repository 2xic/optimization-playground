import torch.nn as nn
from EncoderBlock import EncoderBlock
from PositionalEncoding import PositionalEncoding
from train_transformer import train_transformer

class EncoderTransformer(nn.Module):
    def __init__(self, tokens, num_layers=4) -> None:
        super().__init__()

        self.embedding_dims = 16
        self.d_model = self.embedding_dims

        self.embeddings = nn.Embedding(
            tokens,
            self.embedding_dims
        )
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
    train_transformer(EncoderTransformer)
