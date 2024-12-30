from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from .base_model import BaseModel
import torch.optim as optim
from ..nlp.DocumentEncoderSequence import get_document_dataset as get_document_dataset_sequence
import torch
from .hosted_model import create_flask_app
from .loss_functions import MinimalCrossEntropyLoss, SimpleContrastiveLoss, NegativeSample, NextTokenPrediction
from .dataloader import get_dataloader, TextDataloader
import sys 
from .checkpoints import Checkpoint

class GptEmbeddings(GptTransformerModel):
    def __init__(self, config: Config) -> None:
        super().__init__(config)

    def forward(self, x):
        assert len(x.shape) == 2
        source = self.embedding(x) + self.pos_encoder(x)
        source = self.dropout(source)
        source = self.transformer_decoder(source)
        source = self.layer_norm(source)
        return source.reshape((x.shape[0], -1))

    def get_embedding(self, docs):
        embeddings = torch.zeros((len(docs), 128))
        try:
            for index, text in enumerate(docs):
                X = get_document_dataset_sequence(
                    self.document_encoder, 
                    [text], 
                    SEQUENCE_LENGTH=self.config.sequence_length
                )
                embeddings[index] = self.forward(X).mean(dim=0)
        except Exception as e:
            print(e)
        return embeddings
    
class EmbeddingModelTwo(BaseModel):
    def __init__(self, loss):
        super().__init__(
            self.__class__.__name__
        )
        self.loss_functions = {
            "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
            "SimpleContrastiveLoss": SimpleContrastiveLoss(),
            "NegativeSample": NegativeSample(),
            "NextTokenPrediction": NextTokenPrediction(),
        }
        self.loss = self.loss_functions[loss]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.checkpoint = Checkpoint()
        self.embedding_size = 2048 * 4

    def train_dataloader(self, dataloader: TextDataloader):
        num_epochs = 64
        config = self.get_config(
            self.document_encoder.size,
            2048,
        )
        self.model =    GptTransformerModel(config) \
                        if dataloader.variant == "next_token_prediction" \
                        else GptEmbeddings(config)
        self.optimizer = optim.Adam(self.model.parameters())
        self.model = self.model.to(self.device)
        self.model.train()
        for epoch in range(num_epochs):
            batch = 0
            for (X, y) in dataloader:
                if y is not None:
                    X, y = X.to(self.device), y.to(self.device)
                else:
                    X = X.to(self.device)
                outputs = self.model(X)
                loss = self.loss(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                line = f"Epoch {epoch+1}/{num_epochs}, Batch {batch} Loss: {loss.item():.4f}"
                sys.stdout.write(f"\r{line}")
                sys.stdout.flush()
                batch += 1

                if self.checkpoint.checkpoint():
                    self.save()

        print("Training completed!")
        return self
    
    def _load_model(self, model_data):
        return GptEmbeddings(
            self.get_config(
                model_data["vocab_size"],
                model_data["sequence_length"],
            )
        )
    
    def _get_state_dict(self):
        return {
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.config.vocab_size,
            "sequence_length": self.model.config.sequence_length,
        }

    @classmethod
    def get_config(self, vocab_size, sequence_length):
        embedding_layers = 4
        return Config(
            vocab_size=vocab_size,
            embedding_dim=embedding_layers,
            transformer_layers=2,
            attention_heads=4,
            dropout=0.05,
            feed_forward=8 * embedding_layers,
            padding_index=-1,
            sequence_length=sequence_length
        )

def train_gpt_like():
    model = EmbeddingModelTwo(
        "NextTokenPrediction",
    )
    loader = get_dataloader(
        model.document_encoder,
        variant="next_token_prediction"
    )
    model.train_dataloader(loader)
    return model

def train_embedding():
    model = EmbeddingModelTwo(
        "NegativeSample",
    )
    loader = get_dataloader(
        model.document_encoder,
        variant="sequence"
    )
    model.train_dataloader(loader)
    return model

if __name__ == "__main__":
#    model = train_gpt_like()
    model = train_embedding()
    model.save()
    model.load()
    print(model.transforms([
        "hello, this is some text"
    ]))
    create_flask_app(model).run(
        port=8081,
        host="0.0.0.0"
    )
