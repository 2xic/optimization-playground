from torch.multiprocessing import Process, Lock, Queue
import torch.multiprocessing as mp
from train_byol_model import train as train_byol
from train_model_without_features import train as train_no_features
from train_model_with_random_features import train as train_with_random_features
from train_model_with_byol_features import train_byol_transfer_features
from logs.plot import plot_it


def train_byol_full(lock):
    train_byol(lock)
    train_byol_transfer_features()

if __name__ == "__main__":
    mp.set_start_method('spawn')

    models = [
        train_byol_full,
     #   train_no_features,
        train_with_random_features,
    ]

    lock = Lock()
    processes = []
    for train_func in models:
        processes.append(Process(target=train_func, args=(lock, )))
        processes[-1].start()

    for i in processes:
        i.join()

    plot_it()
