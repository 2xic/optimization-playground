## [What is Late Chunking?](https://weaviate.io/blog/late-chunking)
[Tweet](https://x.com/helloiamleonie/status/1832011929201619442)

- [Naive Chunking](https://weaviate.io/blog/late-chunking#naive-chunking)
  - Chunk the document embeddings for each sentence for instance
  - This looses context
- [Late interaction](https://weaviate.io/blog/late-chunking#late-interaction-and-colbert)
  - Requires more storage (naive approach takes 1/500th of the requirements in the example)
  - We first retrieve the top chucks and then compare the query to the chunks.
- [Late Chunking](https://weaviate.io/blog/late-chunking#late-interaction-and-colbert)
  - Embed the entire document first and then chunk the embeddings.
  - 


## [Hamming Distance Metric Learning](http://www.cs.toronto.edu/~rsalakhu/papers/hamm_distance_metric_learning.pdf)
[paper discussion page](https://wiki.math.uwaterloo.ca/statwiki/index.php?title=hamming_Distance_Metric_Learning)

Learns Hamming distance metrics which learns from a discrete input space into binary codes. 
- Asymmetric Hamming distance
- Probably won't learn any meaningful embedding though given the embedding space would be binary codes, but it allows faster KNN classification times.
