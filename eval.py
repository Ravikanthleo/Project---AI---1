"""Step 5: evaluate retrieval + answers.

Usage (from the folder containing rag.py and eval_questions.json):
    python eval.py --label baseline
    python eval.py --label k8 --k 8
    python eval.py --label retrieval_only --retrieval-only     (fast, no API calls)
"""
import argparse
import csv
import json
import re
from datetime import datetime
from pathlib import Path

import rag

QUESTIONS_FILE = Path("eval_questions.json")
OUT_DIR = Path("eval_results")
REFUSAL_MARKER = "don't have enough information"

JUDGE_PROMPT = """You are grading a documentation Q&A system.

Question: {question}

Context passages the system was given:
{context}

System answer:
{answer}

Score the answer from 1 to 5 on each:
- faithfulness: every claim is supported by the context passages (5 = fully supported, 1 = mostly unsupported or invented).
- relevance: the answer directly and usefully addresses the question (5 = fully, 1 = off-topic).

Reply with ONLY a JSON object: {{"faithfulness": <1-5>, "relevance": <1-5>, "reason": "<one short sentence>"}}"""


def judge(question, answer, retrieved):
    context = rag.build_context(retrieved)[:12000]
    msg = rag._anthropic().messages.create(
        model=rag.MODEL,
        max_tokens=300,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(
            question=question, context=context, answer=answer)}],
    )
    text = "".join(b.text for b in msg.content if b.type == "text")
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def rank_of(url, retrieved):
    for i, c in enumerate(retrieved, start=1):
        if c["url"] == url:
            return i
    return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--retrieval-only", action="store_true")
    args = ap.parse_args()

    questions = json.loads(QUESTIONS_FILE.read_text(encoding="utf-8"))

    # sanity check: do the expected pages exist in the index?
    indexed = {m["url"] for m in rag._collection().get(include=["metadatas"])["metadatas"]}
    missing = sorted({q["url"] for q in questions if q["url"] and q["url"] not in indexed})
    if missing:
        print("WARNING: expected pages not in your index (fix eval_questions.json):")
        for u in missing:
            print("  ", u)

    rows = []
    for i, item in enumerate(questions, start=1):
        q, url = item["q"], item["url"]
        if args.retrieval_only:
            retrieved, answer, judged = rag.retrieve(q, args.k), None, None
        else:
            r = rag.ask(q, args.k)
            retrieved, answer = r["retrieved"], r["answer"]
            judged = None
            if url and REFUSAL_MARKER not in answer.lower():
                judged = judge(q, answer, retrieved)
        refused = bool(answer) and REFUSAL_MARKER in answer.lower()
        row = {"q": q, "expected_url": url, "rank": rank_of(url, retrieved) if url else None,
               "refused": refused, "answer": answer, "judge": judged,
               "retrieved": [{"n": c["n"], "score": c["score"], "url": c["url"], "heading": c["heading"]}
                             for c in retrieved]}
        rows.append(row)
        tag = "OOS" if not url else (f"rank {row['rank']}" if row["rank"] else "MISS")
        print(f"[{i:02d}/{len(questions)}] {tag:7s} {q}")

    inq = [r for r in rows if r["expected_url"]]
    oos = [r for r in rows if not r["expected_url"]]
    hits = [r for r in inq if r["rank"]]
    summary = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "label": args.label,
        "k": args.k,
        "in_scope": len(inq),
        f"hit@{args.k}": round(len(hits) / len(inq), 3),
        "hit@1": round(sum(1 for r in inq if r["rank"] == 1) / len(inq), 3),
        "mrr": round(sum(1 / r["rank"] for r in hits) / len(inq), 3),
    }
    if not args.retrieval_only:
        summary["false_refusals"] = sum(1 for r in inq if r["refused"])
        summary["oos_refused"] = f"{sum(1 for r in oos if r['refused'])}/{len(oos)}"
        summary["avg_faithfulness"] = mean([r["judge"]["faithfulness"] for r in inq if r["judge"]])
        summary["avg_relevance"] = mean([r["judge"]["relevance"] for r in inq if r["judge"]])

    OUT_DIR.mkdir(exist_ok=True)
    (OUT_DIR / f"{args.label}.json").write_text(
        json.dumps({"summary": summary, "rows": rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = OUT_DIR / "summary.csv"
    new = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
        cols = ["time", "label", "k", "in_scope", f"hit@{args.k}", "hit@1", "mrr", "false_refusals",
                "oos_refused", "avg_faithfulness", "avg_relevance"]
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", restval="")
        if new:
            w.writeheader()
        w.writerow(summary)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(f"{k:18s} {v}")
    misses = [r for r in inq if not r["rank"]]
    if misses:
        print("\nRetrieval misses (expected page not in top-k):")
        for r in misses:
            print(f"  - {r['q']}\n      expected {r['expected_url']}\n      got      {r['retrieved'][0]['url']}")
    low = [r for r in inq if r["judge"] and min(r["judge"]["faithfulness"], r["judge"]["relevance"]) <= 3]
    if low:
        print("\nLow-scoring answers:")
        for r in low:
            print(f"  - {r['q']}  ->  {r['judge']}")
    print(f"\nSaved to {OUT_DIR / (args.label + '.json')}")


if __name__ == "__main__":
    main()
