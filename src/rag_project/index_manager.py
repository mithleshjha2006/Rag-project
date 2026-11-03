import json
import logging
from pathlib import Path

from .bm25_indexer import BM25Indexer
from .chunker import DocumentChunker
from .config import BM25_INDEX_PATH, CHUNK_MAP_PATH, FAISS_INDEX_PATH
from .embedder import EmbeddingModel
from .schemas import Chunk, Document
from .vector_indexer import VectorIndexer

logger = logging.getLogger(__name__)


class IndexManager:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.embedder = EmbeddingModel()
        self.bm25_indexer = BM25Indexer()
        self.vector_indexer = VectorIndexer()
        self.chunks: list[Chunk] = []

    # ---------- chunk file ----------

    def save_chunks(self) -> None:
        CHUNK_MAP_PATH.parent.mkdir(parents=True, exist_ok=True)
        data = [chunk.model_dump() for chunk in self.chunks]
        CHUNK_MAP_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load_chunks(self) -> None:
        if not CHUNK_MAP_PATH.exists():
            raise FileNotFoundError(
                f"Chunk file not found: {CHUNK_MAP_PATH}. Run build() first."
            )
        data = json.loads(CHUNK_MAP_PATH.read_text(encoding="utf-8"))
        self.chunks = [Chunk.model_validate(item) for item in data]

    # ---------- build and load ----------

    def build(self, documents: list[Document]) -> None:
        self.chunks = self.chunker.chunk_all(documents)
        if not self.chunks:
            raise ValueError("No chunks produced, nothing to index")

        embeddings = self.embedder.embed_chunks(self.chunks)

        self.bm25_indexer.build(self.chunks)
        self.vector_indexer.build(embeddings)

        self.save_chunks()
        self.embedder.save(embeddings)
        self.bm25_indexer.save()
        self.vector_indexer.save()

        logger.info("Built indexes for %d chunks", len(self.chunks))

    def load(self) -> None:
        self.load_chunks()
        self.bm25_indexer.load()
        self.vector_indexer.load()

        n_chunks = len(self.chunks)
        n_faiss = self.vector_indexer.size
        n_bm25 = self.bm25_indexer.bm25.corpus_size

        if not (n_chunks == n_faiss == n_bm25):
            raise ValueError(
                f"Indexes are out of date: chunks={n_chunks}, "
                f"faiss={n_faiss}, bm25={n_bm25}. Rebuild with build()."
            )

        logger.info("Loaded indexes for %d chunks", n_chunks)

    def is_built(self) -> bool:
        return all(
            Path(p).exists()
            for p in (CHUNK_MAP_PATH, BM25_INDEX_PATH, FAISS_INDEX_PATH)
        )


if __name__ == "__main__":
    from .document_processor import DocumentProcessor

    logging.basicConfig(level=logging.INFO)

    manager = IndexManager()
    if not manager.is_built():
        docs = DocumentProcessor().process_all()
        manager.build(docs)
    else:
        print("Indexes already built, skipping build")

    # the real test: a fresh manager that only loads
    fresh = IndexManager()
    fresh.load()

    print("chunks:", len(fresh.chunks))
    print("faiss :", fresh.vector_indexer.size)
    print("bm25  :", fresh.bm25_indexer.bm25.corpus_size)

    print(fresh.chunks[0].chunk_id, "|", fresh.chunks[0].text[:100])
    print(fresh.chunks[100].chunk_id, "|", fresh.chunks[100].text[:100])