from pathlib import Path

import faiss
import numpy as np

from .config import FAISS_INDEX_PATH


class VectorIndexer:
    def __init__(self):
        self.index = None

    @property
    def size(self) -> int:
        return self.index.ntotal if self.index is not None else 0

    def build(self, embeddings: np.ndarray) -> None:
        if embeddings.ndim != 2:
            raise ValueError(f"Expected a 2D array, got shape {embeddings.shape}")

        embeddings = np.ascontiguousarray(embeddings, dtype="float32")
        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

    def save(self, path: str | Path = FAISS_INDEX_PATH) -> None:
        if self.index is None:
            raise ValueError("Nothing to save: call build() first")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def load(self, path: str | Path = FAISS_INDEX_PATH) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"FAISS index not found: {path}. Build and save it first.")
        self.index = faiss.read_index(str(path))


if __name__ == "__main__":
    from .chunker import DocumentChunker
    from .document_processor import DocumentProcessor
    from .embedder import EmbeddingModel

    docs = DocumentProcessor().process_all()
    chunks = DocumentChunker().chunk_all(docs)

    embedder = EmbeddingModel()
    embeddings = embedder.embed_chunks(chunks)
    print("chunks:", len(chunks), "| embeddings:", embeddings.shape)

    indexer = VectorIndexer()
    indexer.build(embeddings)
    print("index size:", indexer.size)
    indexer.save()

    fresh = VectorIndexer()
    fresh.load()

    # sanity check: a chunk's own vector should find itself first
    scores, positions = fresh.index.search(embeddings[5].reshape(1, -1), 3)
    print("self-search:", positions[0], scores[0].round(3))

    # real searches
    for query in ["What is a recursive neural network?", "label bias"]:
        q = embedder.embed_query(query).reshape(1, -1)
        scores, positions = fresh.index.search(q, 3)

        print("=" * 60)
        print("QUERY:", query)
        for score, pos in zip(scores[0], positions[0]):
            c = chunks[pos]
            print(round(float(score), 3), "|", c.chunk_id, "| page", c.page)
            print("   ", c.text[:200].replace("\n", " "))