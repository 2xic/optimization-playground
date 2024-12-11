from torch.utils.data import DataLoader
from ..training_loops.TrainingLoopAccumulate import TrainingLoopAccumulate

"""
TODO: not good implementation, improve it. 

Also what about when training on multiple GPUs - how to nicely spilt model across devices also ? 
"""
def find_batch_size(trainer: TrainingLoopAccumulate, original_dataloader: DataLoader, device, max_size=128):
    previous_batch_size = original_dataloader.batch_size
    dataloader = None
    while previous_batch_size < max_size:
        new_batch_size = previous_batch_size * 2
        print("new_batch_size ", new_batch_size)
        try:
            dataloader = _copy_data_loader(original_dataloader, new_batch_size)
            _ = trainer._forward(iter(dataloader))
            X, _ = next(iter(dataloader))
            if X.shape[0] < new_batch_size:
                break
        except Exception as e:
            print(e)    
            break
        previous_batch_size = new_batch_size
    finale_size = int(previous_batch_size )#* 0.8)
    print(("previous batch size ", finale_size))
    dataloader = _copy_data_loader(original_dataloader, finale_size)
    return dataloader

def _copy_data_loader(previous: DataLoader, batch_size):
    return DataLoader(
        dataset=previous.dataset,
        sampler=previous.sampler,
        collate_fn=previous.collate_fn,
        drop_last=previous.drop_last,
        # only new field
        batch_size=batch_size,
    )
