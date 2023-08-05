from get_tokenized_shellcode import get_dataloader
from models.TfIdf import TfIdfModel
from eval import get_2d_embeddings, get_cluster_count
from get_tokenized_shellcode import get_dataloader
from models.TransformerEncoder import TransformerEncoderModel, ModelWrapper
from models.TransformerEncoderCompressor import TransformerEncoderCompressor, ModelWrapperEncoder
import torch
import json


def train():
    print(f"Preparing dataset")
    dataloader, dataset = get_dataloader()
    print(f"Preparing training")
    models = [
        ('tf-idf', TfIdfModel()),
        ('transformer', ModelWrapper(TransformerEncoderModel(
            n_token=dataset.n_tokens,
            sequence_size=dataset.sequence_size,
            device=torch.device('cpu'),
            n_layers=1,
        ), epochs=10)),
        ('transformer-encoder-compressor',  ModelWrapperEncoder(TransformerEncoderCompressor(
            n_token=dataset.n_tokens,
            sequence_size=dataset.sequence_size,
            device=torch.device('cpu'),
            n_layers=1,
        ), epochs=10)),
    ]
    for (model_name, model) in models:
        print(f"Training {model_name}")
        """
        All models should have 
            - .train
            - .predict
        """
        model.train(
            dataloader=dataloader,
            dataset=dataset
        )
        predictions = model.predict(
            dataloader=dataloader,
            dataset=dataset
        )
        rows = get_2d_embeddings(predictions)
        clusters_count = get_cluster_count({
            "rows": rows,
        })
        print(f"Clusters count {clusters_count}")
        with open(f"dump_{model_name}.json", "w") as file:
            json.dump({
                "rows": rows,
                "name": model_name,
            }, file)


if __name__ == "__main__":
    train()
