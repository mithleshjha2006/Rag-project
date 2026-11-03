import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import BM25_INDEX_PATH
from .schemas import Chunk
from .tokenizer import tokenize


class BM25Indexer:
    def __init__(self):
        self.bm25 = None

    def build(self, chunks: list[Chunk]) -> None:
        tokenized_corpus = [tokenize(chunk.text) for chunk in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def save(self, path: str | Path = BM25_INDEX_PATH) -> None:
        if self.bm25 is None:
            raise ValueError("Nothing to save: call build() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.bm25, f)

    def load(self, path: str | Path = BM25_INDEX_PATH) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"BM25 index not found: {path}. Build and save it first.")
        with open(path, "rb") as f:
            self.bm25 = pickle.load(f)


if __name__ == "__main__":
    import numpy as np

    from .chunker import DocumentChunker
    from .document_processor import DocumentProcessor

    docs = DocumentProcessor().process_all()
    chunks = DocumentChunker().chunk_all(docs)

    indexer = BM25Indexer()
    indexer.build(chunks)
    indexer.save()

    # new object, to prove the save and load round trip works
    fresh = BM25Indexer()
    fresh.load()

    for query in ["What is a recursive neural network?", "label bias"]:
        tokens = tokenize(query)
        scores = fresh.bm25.get_scores(tokens)
        top = np.argsort(scores)[::-1][:3]

        print("=" * 60)
        print("QUERY:", query, "->", tokens)
        for i in top:
            c = chunks[i]
            print(round(float(scores[i]), 2), "|", c.chunk_id, "| page", c.page)
            print("   ", c.text[:200].replace("\n", " "))