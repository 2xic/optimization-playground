from ..nlp.DocumentEncoderSequence import SimpleVocab, get_document_dataset
import os
import torch

class BaseModel:
    def __init__(self):
        self.document_encoder = SimpleVocab()
        self.model = None

    def fit_transforms(self, docs):
        self.train(docs)
        return self.get_embedding(docs)

    def transforms(self, docs):
        return self.get_embedding(docs)

    def get_embedding(self, docs):
        embeddings = torch.zeros((len(docs), 512))
        for index, text in enumerate(docs):
            inputs = get_document_dataset(self.document_encoder, [text], SEQUENCE_LENGTH=512)
            embedding = torch.zeros((512))
            batch_size = 512
            self.model.eval()
            with torch.no_grad():
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
            "model.pth"
        ))
        self.document_encoder.save(path)
    
    def load(self):
        path = self._get_path()
        os.makedirs(path, exist_ok=True)
        model_data = torch.load(
            os.path.join(
                path,
                "model.pth"
            )
        )
        self.model = self._load_model(model_data)
        self.model.load_state_dict(model_data["state_dict"])
        self.model.eval()
        self.document_encoder = self.document_encoder.load(path)

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
