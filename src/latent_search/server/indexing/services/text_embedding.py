import logging

import torch
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

VECTOR_DIM = 1024


class TextEmbeddingService:
    """
    Service for generating text embeddings using BGE Large EN v1.5.

    Uses BAAI/bge-large-en-v1.5, a model trained specifically for semantic
    text retrieval (STS, NLI, retrieval tasks). Outputs 1024-dimensional
    vectors, matching our existing Qdrant schema.

    Significantly outperforms CLIP-text on text-text matching tasks,
    especially for factual queries involving locations, dates, and
    proper nouns.
    """

    def __init__(
        self,
        model_id: str = "BAAI/bge-large-en-v1.5",
    ):
        self.model_id = model_id
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info(f"Loading text embedding model '{self.model_id}'")
            self._model = SentenceTransformer(self.model_id)
        return self._model

    def encode(self, text: str) -> list[float]:
        """
        Generates a 1024-dimensional embedding for the given text.

        Works for both document/caption text and search queries —
        BGE doesn't distinguish between query and passage encoding.
        """
        with torch.no_grad():
            embedding = self.model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
