[https://python.langchain.com/en/latest/index.html](https://python.langchain.com/en/latest/index.html)

Played a bit around with it, quite cool. Makes it a lot easier to deal with prompts, and history.

My biggest question was regarding `VectorstoreIndexCreator` or how the vector databases has the relationship with the llm.
-> So embeddings are from OpenAi
-> stored in Chroma
-> oh, they just run the query against the vector database ? oki
    -> https://github.com/hwchase17/langchain/blob/928cdd57a4531e606f7ca7e34c0b96736ffcce49/langchain/indexes/vectorstore.py#L33
    -> https://github.com/hwchase17/langchain/blob/928cdd57a4531e606f7ca7e34c0b96736ffcce49/langchain/chains/retrieval_qa/base.py#L82
    ->     search_type: str = "similarity"
    -> https://github.com/hwchase17/langchain/blob/928cdd57a4531e606f7ca7e34c0b96736ffcce49/langchain/vectorstores/pinecone.py#L95

