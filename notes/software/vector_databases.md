## [https://www.pinecone.io/learn/faiss-tutorial/](https://www.pinecone.io/learn/faiss-tutorial/)
- [https://github.com/facebookresearch/faiss](https://github.com/facebookresearch/faiss)
  - Should take a loot at this
- [Faiss - Introduction to Similarity Search](https://youtu.be/sKyvsdEv6rk)
  - Pretty good video
  - Index is split into Voronoi cells
    - Basically k-means ? 
    - You have centroids, and check the query vector against the centroid vector. 
      Then we check against all vectors closest to the closest centroid against the query vector.
  - Index IVF
    - Allow you to search `n` additional centroids that are close
  - Product Quantization
    - Splitting vectors into subvectors
      - Then you cluster all of them nodes into smaller nodes

## [https://weaviate.io](https://weaviate.io)
- [Hierarchical Navigable Small World](https://arxiv.org/ftp/arxiv/papers/1603/1603.09320.pdf)
  - ["HNSW algorithm creates a series of layers. At query time, the HNSW algorithm uses the layers to build a list of approximate nearest neighbors (ANN) quickly and efficiently."](https://weaviate.io/developers/weaviate/concepts/vector-index?utm_source=ainews&utm_medium=email&utm_campaign=ainews-dbrx-best-open-model-but-not-most-efficient#binary-quantization)
