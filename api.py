from contextlib import asynccontextmanager

from fastapi import FastAPI
from pydantic import BaseModel

import rag


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag._collection()   # warm up: load embedding model + DB once at startup
    yield


app = FastAPI(title="FastAPI Docs Q&A", lifespan=lifespan)


class Question(BaseModel):
    question: str
    k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask_endpoint(q: Question):
    r = rag.ask(q.question, q.k)
    return {
        "answer": r["answer"],
        "sources": [
            {
                "n": s["n"],
                "title": s["title"],
                "heading": s["heading"],
                "url": s["url"],
                "snippet": s["text"][:600],
            }
            for s in r["sources"]
        ],
        "invalid_citations": r["invalid_citations"],
    }
