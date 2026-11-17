"""Step 1: load markdown docs into a clean JSONL file.

Each line = one document: {"source", "title", "url", "text"}
Usage: python ingest/load.py
"""
import json
import re
from pathlib import Path

DOCS_DIR = Path("data/fastapi/docs/en/docs")   # where the cloned docs live
OUT_FILE = Path("data/docs.jsonl")
BASE_URL = "https://fastapi.tiangolo.com"

INCLUDE = re.compile(r"\{\*\s*(?:\.\./)*(docs_src/\S+?)(?:\s+[^}]*?)?\s*\*\}")
REPO_DIR = Path("data/fastapi")
MAX_CODE_LINES = 60
stats = {"inlined": 0, "missing": 0}


def inline_code(text: str) -> str:
    """Replace {* ../../docs_src/x.py hl[3] *} placeholders with the real code."""
    def repl(m):
        f = REPO_DIR / m.group(1)
        if not f.exists():
            stats["missing"] += 1
            return ""
        lines = f.read_text(encoding="utf-8").splitlines()[:MAX_CODE_LINES]
        stats["inlined"] += 1
        return "\n```python\n" + "\n".join(lines) + "\n```\n"
    return INCLUDE.sub(repl, text)

def get_title(text: str, fallback: str) -> str:
    """Use the first '# Heading' as the title."""
    match = re.search(r"^#\s+(.+)$", text, flags=re.MULTILINE)
    title = match.group(1).strip() if match else fallback
    return re.sub(r"\s*\{\s*#[^}]*\}\s*$", "", title)   # drop "{ #anchor }"


def to_url(path: Path) -> str:
    """tutorial/first-steps.md -> https://.../tutorial/first-steps/"""
    rel = path.relative_to(DOCS_DIR).with_suffix("")
    if rel.name == "index":
        rel = rel.parent
    return f"{BASE_URL}/{rel.as_posix()}/".replace("/./", "/")


def main() -> None:
    if not DOCS_DIR.exists():
        raise SystemExit(f"Docs not found at {DOCS_DIR}. Did you clone the repo?")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for path in sorted(DOCS_DIR.rglob("*.md")):
            if "release-notes" in path.name:
                continue
            text = path.read_text(encoding="utf-8").strip()
            text = inline_code(text)
            if len(text) < 200:          # skip near-empty pages
                continue
            record = {
                "source": str(path.relative_to(DOCS_DIR)),
                "title": get_title(text, path.stem),
                "url": to_url(path),
                "text": text,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} documents to {OUT_FILE}")
    print(f"Inlined {stats['inlined']} code blocks, {stats['missing']} missing")


if __name__ == "__main__":
    main()
