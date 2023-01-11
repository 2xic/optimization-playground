import torch
import torch.nn as nn


def train_loop(
    X,
    vocab,
    model,
    optimizer,
    epochs=1_000
):
    loss_func = nn.NLLLoss(
        ignore_index=vocab.padding_idx
    )

  #  print(X.shape)
 #   print(vocab.lock())
#    exit(0)
    # torch.nn.CrossEntropyLoss

    SEQ_SIZE = model.SEQ_SIZE
        
    for epoch in range(epochs):
        loss_item = 0

        #for X in [X_0, X_1, X_2, X_3, X_4]:
        for i in range(X.shape[1] - (SEQ_SIZE + 1)):
            optimizer.zero_grad()
            position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) + i
            out = model(X[:, i:i+SEQ_SIZE], position)
            next_token = torch.argmax(out, dim=1)
            expected_next_token = X[:, i + SEQ_SIZE].reshape((X.shape[0]))

            loss = loss_func(
                out,
                expected_next_token,
            )
            loss.backward()
            loss_item += loss.item()
            optimizer.step()
        print(loss_item, epoch)
      
    for i in [X[0], X[1], X[2]]:
        i = i.reshape((1, ) + i.shape)
        for index in range(0, 5):
            position = torch.arange(0, SEQ_SIZE, dtype=torch.long).unsqueeze(0) + index
            out = model(i[:, index:index+SEQ_SIZE], position)
            next_token = torch.argmax(out, dim=1)
            if (0 == index):
                print([vocab.decode(i) for i in i[0, index:index+SEQ_SIZE].tolist()])
            print(f"{vocab.decode(next_token.item())} ({next_token.item()})")
        print("")


