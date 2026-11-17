"""Step 2: chunk docs.jsonl and store embeddings in Chroma.
Usage: python ingest/build_index.py   (safe to re-run; rebuilds from scratch)
"""
import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

from chunking import chunk_document

DOCS_FILE = Path("data/docs.jsonl")
DB_DIR = "chroma_db"
COLLECTION = "fastapi_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"   # small, fast, runs locally
BATCH = 128


def main() -> None:
    docs = [json.loads(l) for l in DOCS_FILE.open(encoding="utf-8")]
    chunks = [c for d in docs for c in chunk_document(d)]
    lengths = sorted(len(c["text"]) for c in chunks)
    print(f"{len(docs)} docs -> {len(chunks)} chunks "
          f"(median {lengths[len(lengths)//2]} chars, max {lengths[-1]})")

    client = chromadb.PersistentClient(path=DB_DIR)
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    col = client.create_collection(COLLECTION, embedding_function=ef,
                                   metadata={"hnsw:space": "cosine"})

    for i in range(0, len(chunks), BATCH):
        b = chunks[i:i + BATCH]
        col.add(ids=[c["id"] for c in b],
                documents=[c["text"] for c in b],
                metadatas=[c["metadata"] for c in b])
        print(f"  indexed {min(i + BATCH, len(chunks))}/{len(chunks)}")
    print(f"Done. {col.count()} chunks stored in ./{DB_DIR}")


if __name__ == "__main__":
    main()
