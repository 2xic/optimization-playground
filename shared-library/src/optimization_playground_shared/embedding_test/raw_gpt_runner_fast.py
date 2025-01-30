"""
Dedicated file without all the wrapper stuff in the "simplified" interface.

Hopefully faster.
"""

from optimization_playground_shared.nlp.GptTransformer import Config, TransformerDecoderWrapper, PositionalEncoding
from optimization_playground_shared.nlp.utils.sampling import temperature_sampling
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
import os
import glob
from dotenv import load_dotenv
import wandb
import shutil
from ..utils.RunHostedModel import ModelHost



load_dotenv()

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

def save_model(model: GptTransformerModel, epoch_number: int, loss):
    state =  {
        "state_dict": model.state_dict(),
        "vocab_size": model.config.vocab_size,
        "sequence_length": model.config.sequence_length,
        "epoch_number": epoch_number,
        "loss": loss,
    }
    # 
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(state, f"checkpoints/{epoch_number}_gpt_raw_fast.pth")
    shutil.copy(f"checkpoints/{epoch_number}_gpt_raw_fast.pth", f"checkpoints/latests_gpt_raw_fast.pth")
    
    return f"checkpoints/{epoch_number}_gpt_raw_fast.pth"

def get_config(vocab_size):
    sequence_length = 256
    embedding_layers = 728 // 2
    return Config(
        vocab_size=vocab_size,
        embedding_dim=embedding_layers,
        transformer_layers=6,
        attention_heads=4,
        dropout=0.05,
        feed_forward=8 * embedding_layers,
        sequence_length=sequence_length,
        padding_index=-1,
    )

def get_vocab():
    document_encoder: BPE = BPE().load(
        "/root/", 
     #   "/home/parallels/", 
        #prefix="pretrained"
    )
    return document_encoder

def train():
    # start a new wandb run to track this script
    run = wandb.init(
        project="my-awesome-project",
        config={
            "learning_rate": 0.02,
        }
    )
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
        batch_size=32
    )
    dataset.SEQUENCE_LENGTH = config.sequence_length

    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    loss_func = NextTokenPrediction(document_encoder.PADDING_IDX)
    progress = tqdm()
    scaler = torch.cuda.amp.GradScaler()
    avg_loss = RunningAverage()
    save_checkpoint = Checkpoint(
        timeout_minutes=120 * 12,
        # Checkpoint every hour.
        checkpoint_numbers=120,
    )
    eval_checkpoint = Checkpoint(
        timeout_minutes=120 * 12,
        # Checkpoint every hour.
        checkpoint_numbers=30,
    )
    output_checkpoint = Checkpoint(
        timeout_minutes=120 * 12,
        # Checkpoint every hour.
        checkpoint_numbers=10,
    )
    text_table = wandb.Table(columns=["epoch", "loss", "X", "y", "y_predicted"])
    for epoch_number in range(1024):
        for (X, y) in dataloader:
            X = X.to(torch.device("cuda:0"))
            y = y.to(torch.device("cuda:1"))
            #with torch.cuda.amp.autocast():
       #     scaler.scale(loss).backward()
       #     scaler.step(optimizer)
       #     scaler.update()
       #     avg_loss.update(loss.item())
            y_predicted = model(X)
            loss = loss_func.forward(y_predicted, y)
            if torch.isnan(loss):
                progress.set_description(f"Loss: nan, skipped")
                continue
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress.set_description(f"Loss: {loss.item():.4f}")
            progress.update(1)

            if output_checkpoint.checkpoint():
                predicted = model(X).tolist()
                text_table.add_data(
                    epoch_number, 
                    loss, 
                    document_encoder.decode(X[0].tolist()),
                    document_encoder.decode(y.reshape((-1)).tolist()[:config.sequence_length]),
                    document_encoder.decode([
                        temperature_sampling(torch.Tensor(i)).item()
                        for i in predicted[:config.sequence_length]
                    ])
                )
                new_text_table = wandb.Table(columns=["epoch", "loss", "X", "y", "y_predicted"])
                for i in text_table.data:
                    new_text_table.add_data(*i)
                run.log({"training_samples" : new_text_table})

            with torch.no_grad():
                if save_checkpoint.checkpoint():
                    name = save_model(model, epoch_number, loss.item())
                    wandb.log({"epoch": epoch_number, "acc": EvaluationMetrics().eval(ModelEmbeddings(name)), "loss": loss.item()})
                else:
                    wandb.log({"epoch": epoch_number, "loss": loss.item()})

            with torch.no_grad():
                if eval_checkpoint.checkpoint():
                    name = save_model(model, epoch_number, loss.item())
                    wandb.log({"epoch": epoch_number, "acc": EvaluationMetrics().eval(ModelEmbeddings(name)), "loss": loss.item()})
                    os.remove(name)
                else:
                    wandb.log({"epoch": epoch_number, "loss": loss.item()})

class ModelEmbeddings:
    def __init__(self, name="gpt_raw_fast.pth"):
        self.document_encoder = get_vocab()
        state = torch.load(name, weights_only=True, map_location=torch.device("cuda:0"))
        config = get_config(state["vocab_size"])
        self.model = GptEmbeddings(config)
        self.model.load_state_dict(state["state_dict"], strict=False)
        self.model.eval()
        del state
        device = torch.cuda.device_count() - 1
        device = torch.device(f"cuda:{device}")
        self.model.to(device)
        self.device = device

    def transforms(self, documents):
        batch_size = 2
        output = None
        for i in range(0, len(documents), batch_size):
            batch_output = self.model.get_embedding(documents[i:i+batch_size], self.document_encoder, self.device)
            if output is None:
                output = batch_output
            else:
                output = torch.concat((
                    output,
                    batch_output
                ), dim=0)
        return output.cpu()

def test(name):
    print(EvaluationMetrics().eval(
        ModelEmbeddings(name)
    ))

def serve():
    host = ModelHost()
    model = ModelEmbeddings("latests_gpt_raw_fast.pth")
    host.add_model("model", model)
    host.run()    

if __name__ == "__main__":
    serve()

    # Loss: 3.1076
    # 
    #print("Hello :=)")
#    train()
#    test()
#    eval_on_dataset()
    # SimpleVocab   : 0.6154
    # BPE           : 0.4872
#    test()
