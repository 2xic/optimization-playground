import chromadb
from chromadb.config import Settings

class Chroma:
    def __init__(self) -> None:
        self.client = chromadb.Client(
            Settings(
                chroma_db_impl="duckdb+parquet",
                persist_directory=".cache/chroma/",
            )
        )
        self.collection = self.client.get_or_create_collection(name="arxiv_test")

    def add_entry(self, id, text, embedding=None):
        try:
            embeddings = embedding if not embedding is None else None
            self.collection.add(
                embeddings=embeddings,
                documents=text,
                ids=id
            )
            print(f"{id} stored")
            self.get_sim_from_id(id)
        except chromadb.errors.IDAlreadyExistsError:
            print(f"{id} is already stored, skipping :)")

    def get_sim_from_id(self, id):
        entry = self.collection.get(ids=[id], include=['embeddings'])
        assert entry["ids"] is not None and len(entry["ids"]) == 1, "ids is none"
        assert entry["embeddings"] is not None, "Embeddings is none"
        return self.get_sim(embedding=entry["embeddings"][0])

    def get_sim(self, embedding):
        output = self.collection.query(
            query_embeddings=embedding,
            n_results=4
        )["ids"]
        return output[0]

    def get_all(self, offset=0):
        # return ids
        #print(self.collection.get())
        #print(self.collection.get(offset=offset, limit=200))
        #print(self.collection.get(offset=offset, limit=200)["ids"])
        return self.collection.get(offset=offset, limit=200)["ids"]
