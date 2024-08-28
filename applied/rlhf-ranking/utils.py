from results import Results, Input
from typing import List
import torch

def rollout_model_binary(model, items: List[Input]) -> List[Results]:
    score: List[Results] = []
    for i in items:
        score.append(Results(
            item_id=i.item_id,
            item_score=0,
            item_tensor=i.item_tensor,
        ))
    with torch.no_grad():
        for _ in range(100):
            i = 1
            swapped = False
            while i < len(items):
                should_swap = model.forward(score[i - 1].item_tensor, score[i].item_tensor)
                if should_swap.shape[-1] == 1:
                    if should_swap.item() < 0.5:
                        score[i - 1], score[i] = score[i], score[i - 1]
                        swapped = True
                elif should_swap[0][0] > should_swap[0][1]:
                    score[i - 1], score[i] = score[i], score[i - 1]
                    swapped = True
                i += 1
            if not swapped:
                break
    # distrusted score
    softmax = torch.nn.Softmax(dim=0)(torch.arange(len(score), 0, -1).float())
    for index in range(len(score)):
        score[index].item_score = softmax[index]
    return score
