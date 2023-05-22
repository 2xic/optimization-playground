# Based off https://pytorch.org/tutorials/beginner/transformer_tutorial.html

from torch.nn import Transformer
from utils import batchify, train_data, val_data, test_data, vocab, generate_square_subsequent_mask, bptt, get_batch, device
import torch
import torch.nn as nn
import time
import math
from model import TransformerModel
from predict import speak

batch_size = 20
eval_batch_size = 10
train_data = batchify(train_data, batch_size)  # shape ``[seq_len, batch_size]``
val_data = batchify(val_data, eval_batch_size)
test_data = batchify(test_data, eval_batch_size)


ntokens = len(vocab)  # size of vocabulary
emsize = 200  # embedding dimension
d_hid = 200  # dimension of the feedforward network model in ``nn.TransformerEncoder``
nlayers = 2  # number of ``nn.TransformerEncoderLayer`` in ``nn.TransformerEncoder``
nhead = 2  # number of heads in ``nn.MultiheadAttention``
dropout = 0.2  # dropout probability

model = Transformer(
   # d_model=ntokens
)

model = TransformerModel(
    ntoken=ntokens,
    d_model=512,
    nhead=nhead,
    d_hid=d_hid,
    nlayers=nlayers,
    dropout=dropout
).to(device)

criterion = nn.CrossEntropyLoss()
lr = 5.0  # learning rate
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)


def train(model: nn.Module) -> None:
    for epoch in range(10):
        model.train()  # turn on train mode
        total_loss = 0.
        log_interval = 200
        start_time = time.time()
        src_mask = generate_square_subsequent_mask(bptt).to(device)

        num_batches = len(train_data) // bptt
        for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
            data, targets = get_batch(train_data, i)
            seq_len = data.size(0)
            if seq_len != bptt:  # only on last batch
                src_mask = src_mask[:seq_len, :seq_len]
            output = model(data, src_mask)
            loss = criterion(output.view(-1, ntokens), targets)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()

            total_loss += loss.item()
            if batch % log_interval == 0 and batch > 0:
                lr = scheduler.get_last_lr()[0]
                ms_per_batch = (time.time() - start_time) * 1000 / log_interval
                cur_loss = total_loss / log_interval
                ppl = math.exp(cur_loss)
                print(f'| epoch {epoch:3d} | {batch:5d}/{num_batches:5d} batches | '
                    f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
                    f'loss {cur_loss:5.2f} | ppl {ppl:8.2f}')
                total_loss = 0
                start_time = time.time()
         #   break
        #break

train(
    model
)
data, targets = get_batch(train_data, 0)
src_mask = generate_square_subsequent_mask(bptt).to(device)
print(
    data
)
speak(
    model,
    data,
    src_mask,
    vocab,
)
# 2 epochs ['=', ',', ',', 'league', 'of', ',', 'and', ',', ',', 'to', ',', ',', '<unk>', 'first', '<unk>', '<unk>', 'she', ',', '<unk>', 'first']
# 10 epcohs ['the', 'of', ',', 'sources', 'irish', ',', 'the', 'in', ',', 'to', ',', ',', 'of', 'first', 'the', 'in', 'the', ',', 'the', 'first']
# -> COOL! 
