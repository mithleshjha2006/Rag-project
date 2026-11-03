"""Retrieval metrics for the evaluation suite.
 
Every metric works on ONE question at a time:
    results  : list of SearchResult, best first
    relevant : list from the golden set, e.g. [{"doc_id": "10_RNTN", "pages": [1, 3]}]
    k        : how many top results to look at
 
Average the per-question values over the golden set to get the final numbers.
"""
import re
 
LIGATURES = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
 
 
def normalise(text: str) -> str:
    for old, new in LIGATURES.items():
        text = text.replace(old, new)
    text = re.sub(r"-\s*\n\s*", "", text)
    text = text.replace("’", "'").replace("“", '"').replace("”", '"')
    return re.sub(r"\s+", " ", text).lower().strip()
 
 
# ---------- relevance helpers ----------
 
def is_relevant(chunk, relevant) -> bool:
    for entry in relevant:
        if chunk.doc_id == entry["doc_id"] and chunk.page in entry["pages"]:
            return True
    return False
 
 
def has_evidence(chunk, evidence) -> bool:
    if not evidence:
        return False
    return normalise(evidence) in normalise(chunk.text)
 
 
# ---------- per-question metrics ----------
 
def hit_at_k(results, relevant, k: int) -> float:
    return 1.0 if any(is_relevant(r.chunk, relevant) for r in results[:k]) else 0.0
 
 
def precision_at_k(results, relevant, k: int) -> float:
    hits = sum(1 for r in results[:k] if is_relevant(r.chunk, relevant))
    return hits / k
 
 
def reciprocal_rank(results, relevant) -> float:
    for rank, r in enumerate(results, start=1):
        if is_relevant(r.chunk, relevant):
            return 1 / rank
    return 0.0
 
 
def coverage_at_k(results, relevant, k: int) -> float:
    targets = {(e["doc_id"], p) for e in relevant for p in e["pages"]}
    if not targets:
        return 0.0
    found = {
        (r.chunk.doc_id, r.chunk.page)
        for r in results[:k]
        if is_relevant(r.chunk, relevant)
    }
    return len(found & targets) / len(targets)
 
 
def evidence_hit_at_k(results, evidence, k: int):
    if not evidence:
        return None
    return 1.0 if any(has_evidence(r.chunk, evidence) for r in results[:k]) else 0.0
 
 
# ---------- averaging ----------
 
def mean(values):
    kept = [v for v in values if v is not None]
    if not kept:
        return None
    return sum(kept) / len(kept)
 
 
# ---------- hand-checked tests ----------
 
if __name__ == "__main__":
    from rag_project.schemas import Chunk, SearchResult
 
    def make(cid, doc, page, text):
        chunk = Chunk(chunk_id=cid, doc_id=doc, paper_title="t", text=text, page=page)
        return SearchResult(chunk=chunk, score=0.0, source="test")
 
    A = make("A", "x", 1, "alpha beta")
    B = make("B", "x", 9, "gamma delta")
    C = make("C", "y", 1, "epsilon")
    rel = [{"doc_id": "x", "pages": [1]}]
 
    def check(name, got, want):
        ok = got is want if want is None else abs(got - want) < 1e-9
        print(("PASS" if ok else "FAIL"), name, "->", got, "(expected", want, ")")
 
    # is_relevant
    check("relevant: right paper+page", float(is_relevant(A.chunk, rel)), 1.0)
    check("relevant: right paper, wrong page", float(is_relevant(B.chunk, rel)), 0.0)
    check("relevant: right page, wrong paper", float(is_relevant(C.chunk, rel)), 0.0)
    check("relevant: empty list", float(is_relevant(A.chunk, [])), 0.0)
 
    # ranking [B, A, C]
    check("hit@1  [B,A,C]", hit_at_k([B, A, C], rel, 1), 0.0)
    check("hit@2  [B,A,C]", hit_at_k([B, A, C], rel, 2), 1.0)
    check("prec@3 [B,A,C]", precision_at_k([B, A, C], rel, 3), 1 / 3)
    check("rr     [B,A,C]", reciprocal_rank([B, A, C], rel), 0.5)
    check("rr     [C,B]", reciprocal_rank([C, B], rel), 0.0)
    check("prec@5 [A,B] (divide by k)", precision_at_k([A, B], rel, 5), 0.2)
 
    # coverage with two relevant places
    rel2 = [{"doc_id": "x", "pages": [1]}, {"doc_id": "y", "pages": [1]}]
    check("coverage@1 [A,B,C]", coverage_at_k([A, B, C], rel2, 1), 0.5)
    check("coverage@3 [A,B,C]", coverage_at_k([A, B, C], rel2, 3), 1.0)
    dup = make("A2", "x", 1, "more text on the same page")
    check("coverage counts a page once", coverage_at_k([A, dup], rel2, 2), 0.5)
 
    # evidence
    check("evidence@1 [A,B]", evidence_hit_at_k([A, B], "gamma", 1), 0.0)
    check("evidence@2 [A,B]", evidence_hit_at_k([A, B], "gamma", 2), 1.0)
    check("evidence None", evidence_hit_at_k([A, B], None, 2), None)
 
    # mean
    check("mean ignores None", mean([1.0, None, 0.0]), 0.5)
    check("mean of nothing", mean([None]), None)
 