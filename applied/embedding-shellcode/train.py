from get_tokenized_shellcode import get_dataloader
from models.TfIdf import TfIdfModel
from eval import get_2d_embeddings, get_cluster_count
from get_tokenized_shellcode import get_dataloader
from models.TransformerEncoder import TransformerEncoderModel, ModelWrapper
import torch

def train():
    dataloader, dataset = get_dataloader()
    models = [
        TfIdfModel(),
        ModelWrapper(TransformerEncoderModel(
            n_token=dataset.n_tokens,
            device=torch.device('cpu'),
            n_layers=1,
        )),
    ]
    for model in models:
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


if __name__ == "__main__":
    train()
