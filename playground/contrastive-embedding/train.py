from dataset import get_tweets
from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from model import EmbeddingLstm
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import random

tweets, users = get_tweets()
tokenieer = Tokenizer()
encoded = torch.tensor(list(map(
    lambda x: tokenieer.encode(x)[0],
    tweets
)))

loss = [

]
unique_users = {
    i:i for i in users
}
unique_users = {
    key: index
    for index, (key, value) in enumerate(unique_users.items())
}

def embedding_loss_func(X, model, tweets):
    tweets_copy = [] + tweets
    random.shuffle(tweets_copy)
    encoded_fuzz = torch.tensor(list(map(
        lambda x: tokenieer.encode_fuzz(x),
        tweets
    )))
    encoded_fuzz_simple = torch.tensor(list(map(
        lambda x: tokenieer.encode_fuzz(x),
        tweets_copy
    )))
    output_a = model(encoded)
    output_b = model(encoded_fuzz)
    output_c = model(encoded_fuzz_simple)

    obj = torch.nn.CosineEmbeddingLoss(
        reduction='sum'
    )
    loss = 0.5 * obj(
        output_a, output_b, 
        -1 * torch.ones((output_a.shape[0]))
    ) + obj(
        output_a, output_c, 
        1 * torch.ones((output_a.shape[0]))
    )
    return loss

def silly_embedding_loss_func(X, model, tweets):
    output_a = model(encoded)
    loss = torch.zeros((0))
    obj = torch.nn.CosineEmbeddingLoss(
        reduction='sum'
    )
    for i in range(output_a.shape[0]):
        for j in range(output_a.shape[0]):
            if i == j:
                pass
            else:
                loss += obj(output_a[i], output_a[j], torch.tensor([-1]))

    return loss * (1 / output_a.shape[0])

def tripley_loss_test(X, model, tweets):
    tweets_copy = [] + tweets
    random.shuffle(tweets_copy)

    encoded_fuzz = torch.tensor(list(map(
        lambda x: tokenieer.encode_fuzz(x),
        tweets
    )))
    encoded_fuzz_simple = torch.tensor(list(map(
        lambda x: tokenieer.encode_fuzz(x),
        tweets_copy
    )))
    output_a = model(encoded)
    output_b = model(encoded_fuzz_simple)
    output_c = model(encoded_fuzz)
    
    loss = torch.max(
        (output_a - output_b) ** 2
        - 
        (output_a - output_c) ** 2,
        0,
        dim=1
    )
    return loss

losses = [
    tripley_loss_test,
    silly_embedding_loss_func,
    embedding_loss_func,
]
figure, axis = plt.subplots(1, len(losses), figsize=(40,20))

for index, loss in enumerate(losses):
    print(loss)
    model = EmbeddingLstm(tokenieer.vocab)
    optimizer = optim.Adam(model.parameters())

    for epoch in range(3_00):  
        loss = embedding_loss_func(
            encoded,
            model,
            tweets
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 25 == 0:
            print(loss)
    print("")

    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(
        model(encoded).detach().numpy()
    )

    axis[index].scatter(
        x=pca_result[:,0],
        y=pca_result[:,1], 
        c=list(map(lambda x: unique_users[x], users))
    )
plt.savefig('plot.png')
print("Saved:)")

