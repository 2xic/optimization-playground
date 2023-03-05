from model import Bob
import torch
import torch.optim as optim

"""
The model seems to never want to output -1,
is there a bug somewhere for this ? 

Yes, there was a bug in the model.
You had applied sigmoid before the tanh in the last layer, so it would never give out -1

Lesson learned ? 

Write small test like this to debug model.
"""
N = 4
DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bob = Bob(
    plaintext=N,
    sharedkey=N
).to(DEVICE)

adam = optim.Adam(bob.parameters())

for step in range(10_000):
    y_pred = None
    if step % 2 == 0:
        X = torch.zeros((100, N)).fill_(1).to(DEVICE)
        y = torch.zeros((100, N)).fill_(1).to(DEVICE)
        y_pred = bob(X, X)
    else:    
        X = torch.zeros((100, N)).fill_(-1).to(DEVICE)
        y = torch.zeros((100, N)).fill_(-1).to(DEVICE)
        y_pred = bob(X, X)

    adam.zero_grad()
#    print(y_pred)
#    print(y)
    error = ((y_pred - y) ** 2).mean()
    error.backward()
    adam.step()
    if step % 100 == 0:
        print(f"Error: {error.item()}")
X = torch.zeros((100, N)).fill_(1).to(DEVICE)
y_pred = bob(X, X)
print(y_pred[:100, :])
X = torch.zeros((100, N)).fill_(-1).to(DEVICE)
y_pred = bob(X, X)
print(y_pred[:100, :])
