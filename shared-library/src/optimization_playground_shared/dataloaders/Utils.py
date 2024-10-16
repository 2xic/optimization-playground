from torch.utils.data import DataLoader
from ..training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate

"""
TODO: not good implementation, improve it. 

Also what about when training on multiple GPUs - how to nicely spilt model across devices also ? 
"""
def find_batch_size(trainer: TrainingLoopAccumulate, original_dataloader: DataLoader, device):
    previous_batch_size = original_dataloader.batch_size
    n = 2
    dataloader = None
    while n < 1_00:
        new_batch_size = previous_batch_size * 2
        print("new_batch_size ", new_batch_size)
        try:
            dataloader = DataLoader(
                dataset=original_dataloader.dataset,
                batch_size=new_batch_size,
                sampler=original_dataloader.sampler,
                collate_fn=original_dataloader.collate_fn,
                drop_last=original_dataloader.drop_last,
            )
            X, y = next(iter(dataloader))
            _ = trainer._forward(X, y)
            if X.shape[0] < new_batch_size:
                break

        except Exception as e:
            print(e)    
            break
        previous_batch_size = new_batch_size
    return dataloader

