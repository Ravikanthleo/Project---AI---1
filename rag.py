"""Step 3: retrieval + Claude answer with verifiable citations.

Usage (from project root):
    python rag.py "How do I declare query parameters?"
    python rag.py --debug "How do I run background tasks?"
"""
import re
import sys
import os

import anthropic
import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv

load_dotenv()  # reads ANTHROPIC_API_KEY from .env

MODEL = "claude-sonnet-5-5"
TOP_K = 5
DB_DIR = "chroma_db"
COLLECTION = "fastapi_docs"
EMBED_MODEL = "all-MiniLM-L6-v2"   # must match build_index.py

RERANK = os.getenv("RERANK", "0") == "1"
CANDIDATES = int(os.getenv("RERANK_CANDIDATES", "20"))
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
EXPAND = os.getenv("EXPAND", "0") == "1"
EXPAND_PAGES = int(os.getenv("EXPAND_PAGES", "2"))
MAX_EXTRA_CHUNKS = int(os.getenv("MAX_EXTRA_CHUNKS", "6"))


def _expand_with_siblings(chunks):
    """Append the other chunks of the top pages (in page order) after the ranked chunks."""
    have = {c["id"] for c in chunks}
    pages = []
    for c in chunks:
        if c["url"] not in pages:
            pages.append(c["url"])
    extra = []
    for url in pages[:EXPAND_PAGES]:
        got = _collection().get(where={"url": url}, include=["documents", "metadatas"])
        sibs = sorted(zip(got["ids"], got["documents"], got["metadatas"]),
                      key=lambda t: int(t[0].rsplit("::", 1)[1]))
        for id_, doc, meta in sibs:
            if id_ in have or len(extra) >= MAX_EXTRA_CHUNKS:
                continue
            extra.append({"id": id_, "text": doc, "score": 0.0, **meta})
            have.add(id_)
    return chunks + extra
_reranker = None


def _get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

SYSTEM_PROMPT = """You answer questions about FastAPI using ONLY the numbered context passages provided.

Rules:
- After every claim taken from a passage, cite it like [1] or [2][3]. Use only the numbers given.
- If the passages do not contain enough information, reply exactly: "I don't have enough information in the docs to answer that." Do not guess or use outside knowledge.
- Keep answers concise. Include short code examples only if they appear in the passages.
- Never invent a citation number."""

_col = None
_client = None


def _collection():
    global _col
    if _col is None:
        ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
        _col = chromadb.PersistentClient(path=DB_DIR).get_collection(COLLECTION, embedding_function=ef)
    return _col


def _anthropic():
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def retrieve(question: str, k: int = TOP_K):
    """Return top-k chunks as dicts, numbered from 1 (optionally reranked / expanded)."""
    n = CANDIDATES if RERANK else k
    res = _collection().query(query_texts=[question], n_results=n)
    chunks = [
        {"id": id_, "text": doc, "score": round(1 - dist, 3), **meta}
        for id_, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0])
    ]
    if RERANK:
        scores = _get_reranker().predict([(question, c["text"]) for c in chunks])
        for c, s in zip(chunks, scores):
            c["score"] = round(float(s), 3)
        chunks.sort(key=lambda c: c["score"], reverse=True)
    chunks = chunks[:k]
    if EXPAND:
        chunks = _expand_with_siblings(chunks)
    for i, c in enumerate(chunks, start=1):
        c["n"] = i
    return chunks


def build_context(chunks) -> str:
    return "\n\n".join(
        f"[{c['n']}] {c['title']} > {c['heading']} ({c['url']})\n{c['text']}" for c in chunks
    )


def extract_citations(answer: str, num_chunks: int):
    """Return (valid_cited_numbers, invalid_numbers) found as [n] in the answer."""
    found = {int(n) for n in re.findall(r"\[(\d+)\]", answer)}
    valid = sorted(n for n in found if 1 <= n <= num_chunks)
    invalid = sorted(n for n in found if not 1 <= n <= num_chunks)
    return valid, invalid


def ask(question: str, k: int = TOP_K) -> dict:
    chunks = retrieve(question, k)
    context = build_context(chunks)
    msg = _anthropic().messages.create(
        model=MODEL,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"Context passages:\n\n{context}\n\nQuestion: {question}",
        }],
    )
    answer = "".join(b.text for b in msg.content if b.type == "text")
    cited, invalid = extract_citations(answer, len(chunks))
    by_n = {c["n"]: c for c in chunks}
    return {
        "answer": answer,
        "sources": [by_n[n] for n in cited],        # only chunks actually cited
        "invalid_citations": invalid,               # should always be empty
        "retrieved": chunks,                        # everything retrieved (for debugging/eval)
    }


if __name__ == "__main__":
    args = sys.argv[1:]
    debug = "--debug" in args
    question = " ".join(a for a in args if a != "--debug") or "How do I declare query parameters?"

    result = ask(question)
    print("\n" + result["answer"] + "\n")

    if result["sources"]:
        print("Sources:")
        for s in result["sources"]:
            print(f"  [{s['n']}] {s['title']} > {s['heading']}\n      {s['url']}")
    if result["invalid_citations"]:
        print(f"\nWARNING: answer cited unknown passages: {result['invalid_citations']}")
    if debug:
        print("\nRetrieved (score, section):")
        for c in result["retrieved"]:
            print(f"  [{c['n']}] {c['score']:.2f}  {c['title']} > {c['heading']}")