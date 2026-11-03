from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME, EMBEDDINGS_PATH
from .schemas import Chunk

QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class EmbeddingModel:
    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME):
        self.model = SentenceTransformer(model_name)
        self.dimension = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            batch_size=32,
            show_progress_bar=True,
        )
        return embeddings.astype(np.float32)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        texts = [chunk.text for chunk in chunks]
        return self.embed_texts(texts)

    def embed_query(self, query: str) -> np.ndarray:
        vectors = self.embed_texts([QUERY_PREFIX + query])
        return vectors[0]

    def save(self, embeddings: np.ndarray, path: str | Path = EMBEDDINGS_PATH) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, embeddings)

    def load(self, path: str | Path = EMBEDDINGS_PATH) -> np.ndarray:
        return np.load(Path(path))