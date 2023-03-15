# from  http://nlp.seas.harvard.edu/annotated-transformer/
import torch
import copy
import torch.nn as nn

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=2):  # 2 = <blank>
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
#        x = torch.argmax(x, dim=2)#.float()
#        x = x.float()
#        #print(x.shape)
#        #print(x.shape)
#        #print(x.shape)
#        #print(y.shape)
#        #print(x.reshape(-1).shape)
#        #print(y.reshape(-1).shape)#
        #print(x.shape)
        #print(y.shape)

        y = y.reshape((-1))
        x = x.reshape((x.shape[-2], x.shape[-1]))
        #print(x.shape)
        #print(y.shape)

        sloss = (nn.CrossEntropyLoss()(
            x, #torch.argmax(x, dim=1).float(),
            y,#.float(),
        ))

     #   sloss = (
     #       self.criterion(
     #            x.reshape(-1), y.reshape(-1)
#                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
     #       #/ norm
     #   )
#        #print(sloss)
#        #print(sloss)
#        #print(sloss)
#        return sloss.data * norm, sloss
        return sloss
    