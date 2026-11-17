"""Try retrieval from the command line.
Usage: python ingest/test_search.py "how do I declare query parameters?"
"""
import sys

import chromadb
from chromadb.utils import embedding_functions

ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
col = chromadb.PersistentClient(path="chroma_db").get_collection("fastapi_docs", embedding_function=ef)

q = " ".join(sys.argv[1:]) or "how do I declare query parameters?"
res = col.query(query_texts=[q], n_results=5)
for doc, meta, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
    print(f"[{1 - dist:.2f}] {meta['title']} > {meta['heading']}\n       {meta['url']}")
