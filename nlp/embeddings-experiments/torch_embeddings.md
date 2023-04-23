### [How does nn.Embedding work?](https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518)
[This](https://discuss.pytorch.org/t/how-does-nn-embedding-work/88518/3) answer seems to suggest that it's just a modified Linear layer. I had problems with that earlier, but I can retry.

The code can be viewed [here](https://pytorch.org/docs/stable/_modules/torch/nn/modules/sparse.html#Embedding). We can see the weights being set like this.
```python
self.weight = Parameter(torch.empty((num_embeddings, embedding_dim), **factory_kwargs)
```
However in the forward it calls `F.embedding` and that calls `torch.embedding`. Which is defined [here](https://github.com/pytorch/pytorch/blob/b85568a54a9c60986235ad1e0cc5dffc71b9d5b1/aten/src/ATen/native/Embedding.cpp#L14)

Hm, what I was mainly looking for was if it was doing a one hot encoding, but does not look like it does that.
