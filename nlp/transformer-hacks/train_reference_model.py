"""
Using https://github.com/karpathy/nanoGPT as a reference to check if there is something wrong with my setup.

Mostly using info from https://github.com/karpathy/nanoGPT/blob/master/train.py
"""
from reference.minigpt_model import GPT, GPTConfig
from train_on_example_text import get_dataset
from tqdm import tqdm
import torch.optim as optim

learning_rate = 6e-4 # max learning rate

tokenizer, dataset = get_dataset()
model = GPT(
    GPTConfig(
        block_size=64,
        vocab_size=dataset.vocab_size,
        n_layer=4,
        n_head=4,
        n_embd=256,
    )
).to('cuda')
#print(model)
#exit(0)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)
dataloader = dataset.iter()

for epoch in range(128):
    for (batch_X, batch_y) in tqdm(dataloader):
        batch_X = batch_X.to('cuda')
        batch_y = batch_y.to('cuda')
        optimizer.zero_grad()
        logits, loss = model(batch_X, batch_y)
        loss.backward()
        optimizer.step()
    print(loss)

X, _ = dataset.sample(1)[0]
INPUT_TOKENS = X.to('cuda').reshape((1, -1))
print(INPUT_TOKENS.shape)
output = model.generate(INPUT_TOKENS, 1024)
print(tokenizer.decode(output[0].tolist()))
