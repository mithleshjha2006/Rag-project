import re

from ..config import RRF_K
from ..schemas import Chunk, SearchResult


class RRFFusion:
    def fuse(self, result_lists: list[list[SearchResult]], k: int = RRF_K) -> list[SearchResult]:
        totals: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}

        for results in result_lists:
            for rank, result in enumerate(results, start=1):
                cid = result.chunk.chunk_id
                totals[cid] = totals.get(cid, 0) + 1 / (k + rank)
                chunks[cid] = result.chunk

        ranked_ids = sorted(totals, key=totals.get, reverse=True)

        return [
            SearchResult(chunk=chunks[cid], score=totals[cid], source="hybrid")
            for cid in ranked_ids
        ]

    def deduplicate(self, results: list[SearchResult]) -> list[SearchResult]:
        seen = set()
        unique = []

        for result in results:
            key = re.sub(r"\s+", " ", result.chunk.text.lower()).strip()[:100]
            if key in seen:
                continue
            seen.add(key)
            unique.append(result)

        return unique


if __name__ == "__main__":
    def make(cid: str) -> SearchResult:
        chunk = Chunk(chunk_id=cid, doc_id="d", paper_title="t",
                      text=f"text of chunk {cid}", page=1)
        return SearchResult(chunk=chunk, score=0.0, source="test")

    A, B, C, D = make("A"), make("B"), make("C"), make("D")
    fusion = RRFFusion()

    # 1. hand-checkable example
    fused = fusion.fuse([[A, B, C], [B, A]])
    for r in fused:
        print(r.chunk.chunk_id, round(r.score, 5), r.source)
    print("expected A and B:", round(1/61 + 1/62, 5), "| C:", round(1/63, 5))

    # 2. a chunk that appears in only one list
    fused = fusion.fuse([[A, B], [D]])
    print([r.chunk.chunk_id for r in fused])

    # 3. one list empty
    fused = fusion.fuse([[], [A, B]])
    print([r.chunk.chunk_id for r in fused])

    # 4. deduplication
    dup = make("E")
    dup.chunk.text = A.chunk.text           # same text as A, different id
    print([r.chunk.chunk_id for r in fusion.deduplicate([A, dup, B])])