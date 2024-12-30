from ..nlp.DocumentEncoderSequence import SimpleVocab, get_document_dataset
import os
import torch
from .checkpoints import Checkpoint

class AccumulateLoss:
    def __init__(self):
        self.counter = 0
        self.loss = None

    def update(self, loss):
        if torch.isnan(loss):
            print("nan loss .... ")
        else:
            if self.loss is None:
                self.loss = loss
            else:
                self.loss += loss
            self.counter += 1

    def reset(self):
        self.counter = 0
        self.loss = None

    def done(self):
        return self.counter >= 32

class BaseModel:
    def __init__(self, name):
        self.document_encoder: SimpleVocab = SimpleVocab()
        self.model = None
        self.name = name
        self.embedding_size = 512
        self.checkpoint = Checkpoint(30)
        self.sequence_length = 2048

    def fit_transforms(self, docs):
        self.train(docs)
        return self.get_embedding(docs)

    def transforms(self, docs):
        return self.get_embedding(docs)

    def get_embedding(self, docs):
        self.model.eval()
        with torch.no_grad():
            return self._get_embedding(docs)
    
    def _get_embedding(self, docs, device=torch.device('cpu')):
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
            )
        )
        self.model = self._load_model(model_data)
        self.model.load_state_dict(model_data["state_dict"])
        self.model.eval()
        self.document_encoder = self.document_encoder.load(
            path,
            self._prefix
        )
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

    def _load_model(self, model_data):
        raise Exception("not implemented")
