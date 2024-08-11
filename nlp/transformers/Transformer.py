import torch.nn as nn
from EncoderBlock import EncoderBlock
from DecoderBlock import DecoderBlock
import torch
from PositionalEncoding import PositionalEncoding
from train_transformer import train_transformer

class Transformer(nn.Module):
    def __init__(self, tokens, num_layers=4) -> None:
        super().__init__()

        self.embedding_dims = 16
        self.d_model = self.embedding_dims

        self.embeddings = nn.Embedding(
            tokens,
            self.embedding_dims
        )
        self.encoder_blocks = nn.ModuleList([
            EncoderBlock(
                self.embedding_dims
            ) for _ in range(num_layers)
        ])
        self.decoder_blocks = nn.ModuleList([
            DecoderBlock(
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

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1). \
            transpose(0, 1)
        mask = mask.float(). \
            masked_fill(mask == 0, float('-inf')). \
            masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, x):
        x = self.embeddings(x)
        y = PositionalEncoding().encode_tensor(x.size(0), d_model=(self.d_model * x.size(0))).reshape((
            x.shape
        ))
        x += y
        encodings = []
        for encoder_block in self.encoder_blocks:
            x = encoder_block(x)
            encodings.append(x)
        # we are allowed to peak
        attn_mask = None # (self.generate_square_subsequent_mask(4))
        
        for index, decoder_block in enumerate(self.decoder_blocks):
            x = decoder_block(x, encodings[index], attn_mask=attn_mask)
        return self.output(x)

if __name__ == "__main__":
    train_transformer(Transformer)
