from dataset import get_tweets
from tokenizer import Tokenizer
import torch
import torch.nn.functional as F
from model import EmbeddingLstm
import torch.optim as optim
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

tweets, users = get_tweets()
tokenieer = Tokenizer()
encoded = torch.tensor(list(map(
    lambda x: tokenieer.encode(x),
    tweets
)))
model = EmbeddingLstm(tokenieer.vocab)
optimizer = optim.Adam(model.parameters())

for _ in range(1_00):
    encoded_fuzz = torch.tensor(list(map(
        lambda x: tokenieer.encode_fuzz(x),
        tweets
    )))
    output_a = model(encoded)
    output_b = model(encoded_fuzz)

    def loss(x, y):
        batch_size = x.shape[0]
        z_i = F.normalize(x, p=2, dim=1)
        z_j = F.normalize(y, p=2, dim=1)

        representations = torch.cat([z_i, z_j], dim=0)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)

        sim_ij = torch.diag(similarity_matrix, batch_size)
        sim_ji = torch.diag(similarity_matrix, -batch_size)

        positives = torch.cat([sim_ij, sim_ji], dim=0)

        mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool)).float()
        temp = 0.05
        nominator = torch.exp(positives / temp)
        denominator = mask.to(similarity_matrix.device) * torch.exp(similarity_matrix / temp)

        all_losses = -torch.log(nominator / torch.sum(denominator, dim=1))
        loss = torch.sum(all_losses) / (2 * batch_size)
        return loss

    loss_value = (loss(output_a, output_b))
        
    optimizer.zero_grad()
    loss_value.backward()
    optimizer.step()
    print(loss_value)
   # break

pca = PCA(n_components=2)
pca_result = pca.fit_transform(
    model(encoded).detach().numpy()
)

ax = plt.figure(figsize=(16,10))
unique_users = {
    i:i for i in users
}
unique_users = {
    key: index
    for index, (key, value) in enumerate(unique_users.items())
}
plt.scatter(
    x=pca_result[:,0],
    y=pca_result[:,1], 
    c=list(map(lambda x: unique_users[x], users))
)
plt.savefig('plot.png')
print("Saved:)")

