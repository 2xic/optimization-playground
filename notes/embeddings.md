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
