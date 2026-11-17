"""Heading-aware chunking for markdown docs."""
import re

MAX_CHARS = 2000       # ~500 tokens
OVERLAP_CHARS = 250    # carried over when a long section is split
MIN_CHARS = 150        # drop tiny fragments

ANCHOR = re.compile(r"\s*\{\s*#[^}]*\}\s*$")


def clean_heading(h: str) -> str:
    return ANCHOR.sub("", h).strip()


def split_sections(text: str):
    """Split markdown into (heading, body) pairs on #/##/### headings.
    Ignores '#' lines inside code fences (Python comments etc.)."""
    sections, heading, buf, in_fence = [], "", [], False
    for line in text.splitlines():
        if line.strip().startswith("```"):
            in_fence = not in_fence
        if not in_fence and re.match(r"^#{1,3}\s+\S", line):
            if buf:
                sections.append((heading, "\n".join(buf).strip()))
            heading = clean_heading(line.lstrip("#"))
            buf = []
        else:
            buf.append(line)
    if buf:
        sections.append((heading, "\n".join(buf).strip()))
    return [(h, b) for h, b in sections if b]


def split_long(body: str):
    """Split an oversized section on blank lines, with overlap."""
    if len(body) <= MAX_CHARS:
        return [body]
    paras, chunks, cur = body.split("\n\n"), [], ""
    for p in paras:
        if cur and len(cur) + len(p) + 2 > MAX_CHARS:
            chunks.append(cur)
            cur = cur[-OVERLAP_CHARS:] + "\n\n" + p   # overlap
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def chunk_document(doc: dict):
    """Yield chunk dicts for one document record from docs.jsonl."""
    title = clean_heading(doc["title"])
    n = 0
    for heading, body in split_sections(doc["text"]):
        for piece in split_long(body):
            if len(piece) < MIN_CHARS:
                continue
            label = f"{title} > {heading}" if heading and heading != title else title
            yield {
                "id": f"{doc['source']}::{n}",
                "text": f"{label}\n\n{piece}",   # heading prefix helps retrieval
                "metadata": {
                    "title": title,
                    "heading": heading or title,
                    "url": doc["url"],
                    "source": doc["source"],
                },
            }
            n += 1
