from train import train
from plot import plot_accuracy_loss

def base_line():
    (epochs, epochs_accuracy, epochs_loss) = train()
    plot_accuracy_loss(
        epochs,
        epochs_accuracy,
        epochs_loss,
        "baseline.png"
    )

if __name__ == "__main__":
    base_line()
