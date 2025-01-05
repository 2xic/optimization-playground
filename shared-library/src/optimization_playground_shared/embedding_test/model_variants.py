import torch
import torch.nn as nn
from ..nlp.DocumentEncoderSequence import SimpleVocab, get_document_dataset
import os 
import abc
from typing import List

class HighLevelModel(abc.ABC):
    model: nn.Module
    
    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def load(self) -> 'HighLevelModel':
        pass

    def transforms(docs: List[str]) -> torch.Tensor:
        pass

"""
Model variants
"""
class BaseModel(HighLevelModel):
    def __init__(self, name, document_encoder: SimpleVocab):
        self.document_encoder: SimpleVocab = document_encoder
        self.model = None
        self.name = name
        self._device = None
        self.embedding_size = 512
        self.sequence_length = 2048

    def fit_transforms(self, docs):
        self.train(docs)
        return self.get_embedding(docs)

    def transforms(self, docs):
        return self.get_embedding(docs, self._device)

    def get_embedding(self, docs, device=torch.device('cpu')):
        self.model.eval()
        with torch.no_grad():
            return self._get_embedding(docs, device)
    
    def _get_embedding(self, docs, device):
        assert type(docs) == list
        embeddings = torch.zeros((len(docs), self.embedding_size), device=device)
        for index, text in enumerate(docs):
            inputs = get_document_dataset(self.document_encoder, [text], SEQUENCE_LENGTH=self.sequence_length)
            inputs = inputs.to(device)
            embedding = torch.zeros((self.embedding_size), device=device)
            batch_size = self.embedding_size
            for i in range(0, inputs.shape[0], batch_size):
                output = self.model(inputs[i:i+batch_size]).mean(dim=0)
                embedding += output
            embeddings[index] = (embedding / inputs.shape[0])
        return embeddings

    def save(self):
        path = self._get_path()
        os.makedirs(path, exist_ok=True)
        torch.save(self._get_state_dict(), os.path.join(
            path,
            self._prefix + "model.pth"
        ))
        self.document_encoder.save(path, self._prefix)
    
    def load(self):
        path = self._get_path()
        os.makedirs(path, exist_ok=True)
        model_data = torch.load(
            os.path.join(
                path,
                self._prefix + "model.pth"
            ),
            weights_only=True
        )
        self.model = self._load_model(model_data)
        self.model.load_state_dict(model_data["state_dict"])
        self.model.eval()
        self.document_encoder = self.document_encoder.load(
            path,
            self._prefix
        )
        self.model.document_encoder = self.document_encoder
        return self

    @property
    def _prefix(self):
        return self.name

    def _get_path(self):
        return os.path.join(os.path.dirname(
            __file__
        ), ".model")
    
    def _get_state_dict(self):
        return {
            "state_dict": self.model.state_dict(),
        }

    def _load_model(self, model_data) -> nn.Module:
        raise Exception("not implemented")


class SimpleEmbeddingModel(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Sequential(*[
             nn.Linear(embed_dim, 1024),
             nn.ReLU(),
             nn.Linear(1024, 2048),
             nn.ReLU(),
             nn.Linear(2048, 4096),
             nn.ReLU(),
             nn.Linear(4096, 8192),
             nn.ReLU(),
             nn.Linear(8192, 4096),
             nn.ReLU(),
             nn.Linear(4096, 2048),
             nn.ReLU(),
             nn.Linear(2048, 1024),
             nn.ReLU(),
             nn.Linear(1024, 512),
             nn.Tanh(),
        ])

    def forward(self, x):
        embedded = self.embedding(x)
        embedded = embedded.mean(dim=1)
        return self.fc(embedded)


class SimpleEmbeddingModelWrapper(BaseModel):
    def __init__(self, name, document_encoder: SimpleVocab):
        super().__init__(name, document_encoder)

    def create_model(self):
        self.model = SimpleEmbeddingModel(self.document_encoder.size, 64)
        return self

    def _load_model(self, model_data):
        return SimpleEmbeddingModel(
            model_data["vocab_size"],
            model_data["embed_dim"]
        )
    
    def _get_state_dict(self):
         return {
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.vocab_size,
            "embed_dim": self.model.embed_dim,
        }

"""
Transformer
"""
from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from ..nlp.DocumentEncoderSequence import get_document_dataset as get_document_dataset_sequence

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

    def get_embedding(self, docs, document_encoder, device):
        embeddings = torch.zeros((len(docs), 512), device=device)
        for index, text in enumerate(docs):
            X = get_document_dataset_sequence(
                document_encoder, 
                [text], 
                SEQUENCE_LENGTH=self.config.sequence_length
            ).to(device)
            output =  self.forward(X)
            embeddings[index] = output.mean(dim=0)
        return embeddings
    

class GptEmbeddingsFineTuned(GptEmbeddings):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.fine_tuned_layers = nn.Sequential(*[
            nn.Linear(self.config.embedding_dim, 512),
            nn.Linear(512, 128),
            nn.Linear(128, self.config.embedding_dim),
        ])
        self.linear_layer = nn.Sequential(*[
            nn.Linear(512, 1),
            nn.Sigmoid()
        ])

    def forward(self, x):
        assert len(x.shape) == 2
        with torch.no_grad():
            source = self.embedding(x) + self.pos_encoder(x)
            source = self.dropout(source)
            source = self.transformer_decoder(source)
            source = self.layer_norm(source)
        source = self.fine_tuned_layers(source)
        return source.reshape((x.shape[0], -1))

class TransformerModelWrapper(BaseModel):
    def __init__(self, name, vocab):
        super().__init__(name, vocab)

    def transforms(self, docs):
       # assert isinstance(self.model, GptEmbeddings)
        self.model.to(self._device)
        with torch.no_grad():
            if isinstance(self.model, GptEmbeddings):
                assert self.document_encoder is not None
                return self.model.get_embedding(docs, self.document_encoder, device=self._device)
            else:
                model = GptTransformerModel(self.config).to(self._device)
                model.load_state_dict(self.model.state_dict())
                return model.get_embedding(docs, self.document_encoder, device=self._device)
            
    def fine_tune_model(self, replacement: nn.Module):
        replacement.load_state_dict(self.model.state_dict(), strict=False)
        self.model = replacement
        return self

    def _load_model(self, model_data):
        model = GptEmbeddings(
            self.get_config(
                model_data["vocab_size"],
                model_data["sequence_length"],
            )
        )
        self.sequence_length = model_data["sequence_length"]
        self.embedding_size = model_data["sequence_length"] * 4

        return model
    
    def _get_state_dict(self):
        return {
            "state_dict": self.model.state_dict(),
            "vocab_size": self.model.config.vocab_size,
            "sequence_length": self.model.config.sequence_length,
        }

    def create_model(self):
        self.config = self.get_config(self.document_encoder.size, 128)
        self.model = GptTransformerModel(self.config) 

        self.sequence_length = self.config.sequence_length
        return self

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
    