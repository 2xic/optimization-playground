from get_tokenized_shellcode import get_dataloader
from models.TransformerEncoder import TransformerEncoderModel, ModelWrapper
import torch
from optimization_playground_shared.plot.Plot import Plot, Figure

def train():
    dataloader, dataset = get_dataloader()
    models = [
        ModelWrapper(TransformerEncoderModel(
            n_token=dataset.n_tokens,
            device=torch.device('cpu'),
            n_layers=1,
        ))
    ]
    loss_over_time = []
    for model in models:
        for epoch in range(10):
            sum_loss = 0
            for X in dataloader:
                loss = model.fit(X)
                sum_loss += loss.item()
                print(epoch, loss)
            loss_over_time.append(sum_loss)
            torch.save({
                "model": model.model.state_dict(),
            }, 'model.pkt')
            """
            Loss over time
            """
            plot = Plot()
            plot.plot_figures(
                figures=[
                    Figure(
                        plots={
                            "Loss": loss_over_time,
                        },
                        title="Training loss",
                        x_axes_text="Epochs",
                        y_axes_text="Loss",
                    ),
                ],
                name=f'loss.png'
            )
if __name__ == "__main__":
    train()
