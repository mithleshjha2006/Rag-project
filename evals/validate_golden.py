"""Validate evals/data/golden_set.json against YOUR processed documents.

Run from the project root:
    uv run python evals/validate_golden.py

For every question it checks:
  - ids are unique and required fields exist
  - every relevant doc_id exists in data/processed/
  - every relevant page number exists in that document
  - if 'evidence' is set, the phrase appears (after normalisation) on one of the claimed pages;
    if it appears on a different page, it tells you which one so you can fix the label
"""
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLDEN = ROOT / "evals" / "data" / "golden_set.json"
PROCESSED = ROOT / "data" / "processed"

LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
VALID_TYPES = {"exact_term", "paraphrased", "multi_paper", "unanswerable", "adversarial"}


def normalise(text: str) -> str:
    for old, new in LIGATURES.items():
        text = text.replace(old, new)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).lower().strip()


def load_documents() -> dict[str, dict[int, str]]:
    docs = {}
    for path in sorted(PROCESSED.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        docs[data["doc_id"]] = {p["page_number"]: normalise(p["text"]) for p in data["pages"]}
    return docs


def main() -> int:
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    items = golden["items"]
    docs = load_documents()
    if not docs:
        print(f"No processed documents found in {PROCESSED}. Run DocumentProcessor first.")
        return 1

    errors, warnings = [], []
    ids = [it["id"] for it in items]
    for dup, n in Counter(ids).items():
        if n > 1:
            errors.append(f"duplicate id {dup}")

    for it in items:
        qid = it["id"]
        for field in ("question", "type", "expected_answer", "relevant"):
            if field not in it:
                errors.append(f"{qid}: missing field '{field}'")
        if it.get("type") not in VALID_TYPES:
            errors.append(f"{qid}: unknown type {it.get('type')!r}")

        answerable = it.get("type") not in ("unanswerable", "adversarial")
        if answerable and not it.get("relevant"):
            errors.append(f"{qid}: answerable question has no 'relevant' entries")
        if not answerable and it.get("relevant"):
            warnings.append(f"{qid}: {it['type']} question should have an empty 'relevant' list")

        evidence = it.get("evidence")
        claimed_pages = {(r["doc_id"], p) for r in it.get("relevant", []) for p in r["pages"]}

        for rel in it.get("relevant", []):
            doc_id = rel["doc_id"]
            if doc_id not in docs:
                errors.append(f"{qid}: doc_id {doc_id!r} not found in data/processed")
                continue
            for page in rel["pages"]:
                if page not in docs[doc_id]:
                    errors.append(f"{qid}: {doc_id} has no page {page} (pages after the references cut are removed)")

        if evidence:
            needle = normalise(evidence)
            found = [(d, p) for d, pages in docs.items() for p, text in pages.items() if needle in text]
            if not found:
                warnings.append(f"{qid}: evidence not found anywhere: {evidence!r}")
            elif not (set(found) & claimed_pages):
                warnings.append(f"{qid}: evidence found on {found[:4]} but you claimed {sorted(claimed_pages)}")

    print(f"{len(items)} questions, {len(docs)} processed documents")
    print("By type:", dict(Counter(it["type"] for it in items)))
    covered = Counter(r["doc_id"] for it in items for r in it.get("relevant", []))
    print("Questions per paper:", dict(covered))
    for doc_id in docs:
        if doc_id not in covered:
            warnings.append(f"no question points at {doc_id}")

    for w in warnings:
        print("WARN :", w)
    for e in errors:
        print("ERROR:", e)
    print(f"\n{len(errors)} errors, {len(warnings)} warnings")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
