from sentence_transformers import CrossEncoder

from ..config import FINAL_TOP_K, RERANKER_MODEL_NAME
from ..schemas import SearchResult


class Reranker:
    def __init__(self, model_name: str = RERANKER_MODEL_NAME):
        self.model = CrossEncoder(model_name)

    def rerank(
        self, query: str, results: list[SearchResult], top_k: int = FINAL_TOP_K
    ) -> list[SearchResult]:
        if not results:
            return []

        pairs = [(query, r.chunk.text) for r in results]
        scores = self.model.predict(pairs)

        ranked = sorted(zip(results, scores), key=lambda pair: pair[1], reverse=True)

        return [
            SearchResult(chunk=r.chunk, score=float(score), source="reranked")
            for r, score in ranked[:top_k]
        ]


if __name__ == "__main__":
    import time

    from ..index_manager import IndexManager
    from .retrievers import HybridRetriever

    manager = IndexManager()
    manager.load()

    hybrid = HybridRetriever(manager)
    reranker = Reranker()

    for query in ["What is a recursive neural network?", "label bias", "zzzzqx"]:
        print("=" * 70)
        print("QUERY:", query)

        candidates = hybrid.retrieve(query, top_k=20)
        print("--- hybrid top 5 (before reranking)")
        for r in candidates[:5]:
            print(f"  {r.score:.4f} | {r.chunk.chunk_id} | page {r.chunk.page}")

        start = time.time()
        final = reranker.rerank(query, candidates, top_k=5)
        print(f"--- reranked top 5 ({time.time() - start:.2f}s for {len(candidates)} candidates)")
        for r in final:
            print(
                f"  {r.score:.3f} | {r.chunk.chunk_id} | page {r.chunk.page} | "
                f"{r.chunk.text[:100].replace(chr(10), ' ')}"
            )