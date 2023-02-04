from torch.multiprocessing import Process, Lock, Queue
import torch.multiprocessing as mp
from train_byor_model import train as train_byor
from train_model_without_features import train as train_no_features
from train_model_with_random_features import train as train_with_radom_featues
from train_model_with_byol_features import train_byol
from logs.plot import plot_it

if __name__ == "__main__":
    mp.set_start_method('spawn')

    models = [
        train_byor,
#        train_no_features,
#        train_with_radom_featues,
    ]

    lock = Lock()
    processes = []
    for train_func in models:
        processes.append(Process(target=train_func, args=(lock, )))
        processes[-1].start()

    for i in processes:
        i.join()

    train_byol()

    plot_it()
