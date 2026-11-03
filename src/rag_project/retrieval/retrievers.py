import numpy as np

from ..config import BM25_TOP_K, DENSE_TOP_K, RERANK_TOP_N
from ..index_manager import IndexManager
from ..schemas import SearchResult
from ..tokenizer import tokenize
from .fusion import RRFFusion


class BM25Retriever:
    def __init__(self, manager: IndexManager):
        self.chunks = manager.chunks
        self.bm25 = manager.bm25_indexer.bm25

    def retrieve(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        top_positions = np.argsort(scores)[::-1][:top_k]

        results = []
        for pos in top_positions:
            score = float(scores[pos])
            if score <= 0:
                continue
            results.append(SearchResult(
                chunk=self.chunks[int(pos)],
                score=score,
                source="bm25",
            ))
        return results


class DenseRetriever:
    def __init__(self, manager: IndexManager):
        self.chunks = manager.chunks
        self.index = manager.vector_indexer.index
        self.embedder = manager.embedder

    def retrieve(self, query: str, top_k: int = DENSE_TOP_K) -> list[SearchResult]:
        vector = self.embedder.embed_query(query).reshape(1, -1)
        scores, positions = self.index.search(vector, top_k)

        results = []
        for score, pos in zip(scores[0], positions[0]):
            if pos == -1:
                continue
            results.append(SearchResult(
                chunk=self.chunks[int(pos)],
                score=float(score),
                source="dense",
            ))
        return results


class HybridRetriever:
    def __init__(self, manager: IndexManager):
        self.bm25 = BM25Retriever(manager)
        self.dense = DenseRetriever(manager)
        self.fusion = RRFFusion()

    def retrieve(
        self, query: str, top_k: int = RERANK_TOP_N, verbose: bool = False
    ) -> list[SearchResult]:
        if not query.strip():
            return []

        bm25_results = self.bm25.retrieve(query)
        dense_results = self.dense.retrieve(query)

        fused = self.fusion.fuse([bm25_results, dense_results])
        unique = self.fusion.deduplicate(fused)

        if verbose:
            print(
                f"bm25={len(bm25_results)} dense={len(dense_results)} "
                f"fused={len(fused)} after_dedup={len(unique)}"
            )

        return unique[:top_k]


if __name__ == "__main__":
    manager = IndexManager()
    manager.load()

    bm25 = BM25Retriever(manager)
    dense = DenseRetriever(manager)
    hybrid = HybridRetriever(manager)

    def show(label: str, results: list[SearchResult]) -> None:
        print(f"--- {label}: {len(results)} results")
        for r in results:
            print(
                f"  {r.source} | {r.score:.4f} | {r.chunk.chunk_id} | "
                f"page {r.chunk.page} | "
                f"{r.chunk.text[:100].replace(chr(10), ' ')}"
            )

    for query in ["What is a recursive neural network?", "label bias", "zzzzqx", ""]:
        print("=" * 70)
        print("QUERY:", repr(query))
        show("bm25", bm25.retrieve(query, top_k=5))
        show("dense", dense.retrieve(query, top_k=5) if query.strip() else [])
        show("hybrid", hybrid.retrieve(query, top_k=5, verbose=True))