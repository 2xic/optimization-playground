"""
Data source = arxiv ? 
"""
from dotenv import load_dotenv
load_dotenv()

from optimization_playground_shared.datasources.arxiv import start_crawling, get_crawled
from optimization_playground_shared.apis.openai_embeddings import OpenAiEmbeddings
from database import Chroma

model = OpenAiEmbeddings()
database = Chroma()

for i in start_crawling([
    "1911.08265", # muzero paper
    "1707.03497", # value prediction network paper
], limit_downloads=0):
    print(i)

for i in get_crawled():
    embedding = model.get_embedding(i.text)
    assert embedding is not None
    database.add_entry(
        i.id,
        i.text,
        embedding
    )
