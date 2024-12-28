from optimization_playground_shared.nlp.GptTransformer import GptTransformerModel, Config
from .base_model import BaseModel
import torch.optim as optim
from ..nlp.DocumentEncoderSequence import get_document_dataset as get_document_dataset_sequence
import torch
from ..apis.url_to_text import get_url_documents
from .model_1 import MinimalCrossEntropyLoss, SimpleContrastiveLoss, NegativeSample
from .hosted_model import create_flask_app

class EmbeddingModelTwo(BaseModel):
    def __init__(self, loss):
        super().__init__()
        self.loss_functions = {
            "MinimalCrossEntropyLoss": MinimalCrossEntropyLoss(),
            "SimpleContrastiveLoss": SimpleContrastiveLoss(),
            "NegativeSample": NegativeSample(),
        }
        self.loss = self.loss_functions[loss]
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def train(self, docs):
        self.document_encoder.fit(docs)
        self.document_encoder.lock()

        num_epochs = 64
        batch_size = 256

        self.model = GptTransformerModel(self.get_config(
            self.document_encoder.size,
            32,
        ))
        self.optimizer = optim.Adam(self.model.parameters())
        X = get_document_dataset_sequence(
            self.document_encoder, 
            docs, 
            SEQUENCE_LENGTH=self.model.config.sequence_length
        )
        self.model = self.model.to(self.device)
        for epoch in range(num_epochs):
            for i in range(0, X.shape[0], batch_size):
                self.model.train()
                outputs = self._model_raw_forward(X[i:i+batch_size].to(self.device))
                loss = self.loss(outputs)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

        print("Training completed!")
        return self
    
    def _model_raw_forward(self, x):
        assert len(x.shape) == 2
        source = self.model.embedding(x) + self.model.pos_encoder(x)
        source = self.model.dropout(source)
        source = self.model.transformer_decoder(source)
        source = self.model.layer_norm(source)
        return source.reshape((x.shape[0], -1))

    def get_embedding(self, docs):
        embeddings = torch.zeros((len(docs), 128))
        try:
            for index, text in enumerate(docs):
                X = get_document_dataset_sequence(
                    self.document_encoder, 
                    [text], 
                    SEQUENCE_LENGTH=self.model.config.sequence_length
                )
                embeddings[index] = self._model_raw_forward(X).mean(dim=0)
        except Exception as e:
            print(e)
        return embeddings

    def _load_model(self, model_data):
        return GptTransformerModel(
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
        return Config(
            vocab_size=vocab_size,
            embedding_dim=4,
            transformer_layers=4,
            attention_heads=4,
            dropout=0.05,
            feed_forward=8 * 4,
            padding_index=-1,
            sequence_length=sequence_length
        )

if __name__ == "__main__":
    model = EmbeddingModelTwo(
        "NegativeSample",
    ).train(
        get_url_documents(
            pages=50
        )
    )
    model.save()
    model.load()
    print(model.transforms([
        "hello, this is some text"
    ]))
    create_flask_app(model).run(
        port=8081,
        host="0.0.0.0"
    )
