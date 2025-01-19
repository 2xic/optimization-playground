"""
Dedicated file without all the wrapper stuff in the "simplified" interface.

Hopefully faster.
"""

from optimization_playground_shared.nlp.GptTransformer import Config, TransformerDecoderWrapper, PositionalEncoding
import torch.nn as nn
from torch import Tensor
from ..nlp.wordpiece.bpe import BPE
from ..nlp.SimpleVocab import SimpleVocab
from .dataloader import get_dataloader
from .loss_functions import NextTokenPrediction
import torch.optim as optim
from tqdm import tqdm
import torch
from optimization_playground_shared.utils.RunningAverage import RunningAverage
from .checkpoints import Checkpoint
from .model_variants import GptEmbeddings
from .evals import EvaluationMetrics

"""
Copy of the transformer implementation inside optimization_playground_shared.nlp.GptTransformer
"""

class GptTransformerModel(nn.Module):
    def __init__(self, config: Config) -> None:
        super(GptTransformerModel, self).__init__()
        self.config = config

        self.embedding = nn.Embedding(
            config.vocab_size, 
            config.embedding_dim,
            padding_idx=config.padding_index
        )
        layers = []
        for _ in range(config.transformer_layers):
            layers.append(nn.TransformerDecoderLayer(
                d_model=config.embedding_dim, 
                nhead=config.attention_heads, 
                dim_feedforward=config.feed_forward, 
                dropout=config.dropout,
                batch_first=True,
                activation=nn.functional.gelu,
            ))

        self.transformer_decoder = TransformerDecoderWrapper(layers)
        self.layer_norm = nn.LayerNorm(self.config.embedding_dim) 
        self.output = nn.Sequential(*[
            nn.Linear(config.embedding_dim, config.vocab_size, bias=False),
        ])
        self.pos_encoder = PositionalEncoding(
            config.embedding_dim,
            config.dropout,
            max_len=config.sequence_length,
        )
        self.sequence_size = config.sequence_length
        self.dropout = nn.Dropout(config.dropout)

    def to_gpu(self):
        # device 1
        self.embedding.to(torch.device("cuda:0"))
        self.pos_encoder.to(torch.device("cuda:0"))
        self.dropout.to(torch.device("cuda:0"))
        # device 2
        self.transformer_decoder.to(torch.device("cuda:1"))
        self.layer_norm.to(torch.device("cuda:1"))
        self.output.to(torch.device("cuda:1"))
        return self

    def raw_forward(self, x: Tensor):
        assert len(x.shape) == 2
        source = self.embedding(x) + self.pos_encoder(x)
        source = self.dropout(source)
        source = source.to(torch.device("cuda:1"))
        source = self.transformer_decoder(source)
        source = self.layer_norm(source)
        source = self.output(source)
        return source

    def forward(self, x: Tensor):
        results = self.raw_forward(x)
        reshaped = results.view(-1, self.config.vocab_size)
        return reshaped

def save_model(model: GptTransformerModel):
    state =  {
        "state_dict": model.state_dict(),
        "vocab_size": model.config.vocab_size,
        "sequence_length": model.config.sequence_length,
    }
    torch.save(state, "gpt_raw_fast.pth")

def get_config(vocab_size):
    sequence_length = 256
    embedding_layers = 4
    return Config(
        vocab_size=vocab_size,
        embedding_dim=embedding_layers,
        transformer_layers=2,
        attention_heads=4,
        dropout=0.05,
        feed_forward=8 * embedding_layers,
        sequence_length=sequence_length,
        padding_index=-1,
    )

def get_vocab():
    document_encoder: BPE = BPE().load(
        "/root/", 
        prefix="pretrained"
    )
    return document_encoder

def train():
    document_encoder = get_vocab()
#    document_encoder: SimpleVocab = SimpleVocab().load(
#        "/root/", 
#        prefix="pretrained"
#    )
    vocab_size = document_encoder.size
    config = get_config(vocab_size)
    model = GptTransformerModel(config).to_gpu()
    dataloader, dataset = get_dataloader(
        document_encoder, 
        "next_token_prediction",
        32
    )
    dataset.SEQUENCE_LENGTH = config.sequence_length

    optimizer = optim.Adam(model.parameters())
    loss_func = NextTokenPrediction(document_encoder.PADDING_IDX)
    progress = tqdm()
    scaler = torch.cuda.amp.GradScaler()
    avg_loss = RunningAverage()
    checkpoint = Checkpoint(
        timeout_minutes=120 * 12
    )
    for i in range(1024):
        for (X, y) in dataloader:
            X = X.to(torch.device("cuda:0"))
            y = y.to(torch.device("cuda:1"))
            with torch.cuda.amp.autocast():
                y_predicted = model(X)
                loss = loss_func.forward(y_predicted, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            avg_loss.update(loss.item())
            optimizer.zero_grad()

            progress.set_description(f"Loss: {avg_loss.value:.4f}")
            progress.update(1)
            if checkpoint.checkpoint():
                save_model(model)

class ModelEmbeddings:
    def __init__(self):
        self.document_encoder = get_vocab()
        state = torch.load( "gpt_raw_fast.pth", weights_only=True)
        config = get_config(state["vocab_size"])
        self.model = GptEmbeddings(config)
        self.model.load_state_dict(state["state_dict"], strict=False)

    def transforms(self, documents):
        self.model.eval()
        with torch.no_grad():
            device = torch.device("cuda:0")
            self.model.to(device)
            return self.model.get_embedding(documents, self.document_encoder, device)

def test():
    print(EvaluationMetrics().eval(
        ModelEmbeddings()
    ))

if __name__ == "__main__":
#    train()
    test()
    # SimpleVocab   : 0.6154
    # BPE           : 0.4872
#    test()
