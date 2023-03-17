from Attention import scaled_dot_product
import torch.nn as nn
import torch
from utils import Tokenizer
import math
import torch.nn.functional as F
import matplotlib.pyplot as plt


def scaled_dot_product(q, k, v):
    d_k = q.size()[-1]
    attn_logits = torch.matmul(q, k.transpose(-2, -1))
    attn_logits = attn_logits / math.sqrt(d_k)
    attention = F.softmax(attn_logits, dim=-1)
    values = torch.matmul(attention, v)
    return values, attention

# from https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
class MultiHeadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        print(input_dim)
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
      #  print(x.shape)
        qkv = self.qkv_proj(x)

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v)
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return o, attention


class MultiHeadAttentionExample(nn.Module):
    def __init__(self, tokens, sequence_length):
        super().__init__()
        
        self.embedding = nn.Embedding(tokens, 4)#8 * 3)
       # self.embedding = nn.Linear(tokens, 8)
        self.attention = MultiHeadAttention(sequence_length, 64, 64)
        #self.attention = MultiHeadAttention(4 * 8)
        self.sequence_length = sequence_length

        self.fake_attention = nn.Linear(
            16, #tokens * 4,
            64 * 4
        )

        self.middle = nn.Linear(
            64 * 4,
            64 * 4,
        )

        self.output = nn.Linear(
            64 * 4,
            1
        )

    def forward(self, X):
        assert X.size(1) == self.sequence_length
        embed = self.embedding(X)

        #attention_value = embed.reshape((X.size(0), -1))
        #fake_attention = self.fake_attention(attention_value)
        #attention_value = fake_attention
        
        attention_value, attention  = self.attention(embed)
        attention_value = F.sigmoid(attention_value).reshape((X.size(0), -1))

        output = F.sigmoid(self.middle(attention_value))
        output = F.sigmoid(self.output(output))

        return output,attention


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
        0
    ]
    tokenizer = Tokenizer().encode_documents(documents)
    model = MultiHeadAttentionExample(
        len(tokenizer.word_idx),
        sequence_length=4
    )

    optimizer = torch.optim.Adam(model.parameters())
    model.train()

    for _ in range(1_000):
        total_loss = 0
        y_target = torch.tensor([
            [y]
            for y in targets
        ]).float()
        X = torch.concat([
            tokenizer.encode_document_tensor(x, sequence_length=model.sequence_length)
            for x in documents
        ], dim=0)
        y, attn = model(X)

        loss = nn.BCELoss()(
            y,
            y_target,
        )

        optimizer.zero_grad()
        loss.backward()
        total_loss += loss.item()
        optimizer.step()

    model.eval()

    with torch.no_grad():
        fig, axes = plt.subplots(nrows=1, ncols=2)

        for index, (
            title,
            document
        ) in enumerate(
            zip([
                "1 value document",
                "0 value document"
            ],
            [
                documents[0],
                documents[2]
            ])
        ):
            X = tokenizer.encode_document_tensor(document, sequence_length=model.sequence_length).reshape((
                1, -1
            ))
            _, attn = model(X)

            axes[index].set_axis_off()
            axes[index].set_title(title)
            axes[index].imshow(
                attn.numpy().reshape((32, 32))
            )
        plt.savefig('MultiHeadAttention.png')


