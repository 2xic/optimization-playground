import matplotlib.pyplot as plt
import os 
import numpy as np

def running_average(data):
    running_sum = 0
    for i, value in enumerate(data, 1):
        running_sum += value
        running_avg = running_sum / i
        yield running_avg
    
def plot_accuracy_loss(
    epochs,
    accuracy,
    loss,
    file_name: str
):
    fig, ax1 = plt.subplots()

    accuracy = list(running_average(accuracy))

    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(epochs, accuracy, color='tab:blue', label='Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')

    ax2 = ax1.twinx()
    ax2.set_ylabel('Loss', color='tab:red')
    ax2.plot(epochs, loss, color='tab:red', label='Loss')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Training Accuracy and Loss')
    fig.tight_layout()
    file_name = file_name.split(".")[0]
    plt.savefig(f"{file_name}.png")
