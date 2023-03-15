from .helpers import Batch, SimpleLossCompute
from typing import List
from .models.encoder import EncoderDecoder
import torch.nn as nn
import torch.optim as optim
import torch

def train_model(model: EncoderDecoder, batches: List[Batch]):
    loss_compute = SimpleLossCompute(
        model.model.generator,
        torch.nn.L1Loss(),#; .#KLDivLoss(reduction="sum")
    )
    optimizer = optim.Adam(model.model.parameters())

    for _ in range(1_00):
        for (src, target) in batches:
#            print(src)
#            print(target)

            batch = Batch(src, target)
            out = model.model.forward(
                batch.src, batch.tgt, batch.src_mask, batch.tgt_mask
            )
            loss = loss_compute(out, batch.tgt_y, 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

