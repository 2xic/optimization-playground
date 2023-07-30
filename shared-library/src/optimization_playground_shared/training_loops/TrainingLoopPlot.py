from ..plot.Plot import Plot, Figure
from .TrainingLoop import TrainingLoop
from tqdm import tqdm

class TrainingLoopPlot:
    def __init__(self, training_loop: TrainingLoop):
        self.training_loop = training_loop

    def fit(self, dataloader, epochs):
        training_accuracy = []
        training_loss = []

        for _ in tqdm(range(epochs)):
            (loss, accuracy) = self.training_loop.train(dataloader)
            training_accuracy.append(accuracy)
            training_loss.append(loss)
        
        plot = Plot()
        name = self.training_loop.model.__class__.__name__
        plot.plot_figures(
            figures=[
                Figure(
                    plots={
                        "Loss": training_loss,
                    },
                    title="Training loss",
                    x_axes_text="Epochs",
                    y_axes_text="Loss",
                ),
                Figure(
                    plots={
                        "Training accuracy": training_accuracy,
                    },
                    title="Accuracy",
                    x_axes_text="Epochs",
                    y_axes_text="accuracy",
                ),
            ],
            name=f'training_{name}.png'
        )
