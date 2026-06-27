import math
import uuid
from typing import Any

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    SparseVector,
    SparseVectorParams,
    VectorParams,
)

VECTOR_NAMES = ("image", "text")
EXPECTED_DIM = 1024
SPARSE_VECTOR_NAME = "sparse"


def _validate_vector(vec: list[float], name: str) -> None:
    """Raise if the vector has wrong dimensions or contains bad values."""
    if len(vec) != EXPECTED_DIM:
        raise ValueError(
            f"{name} embedding has {len(vec)} dims, expected {EXPECTED_DIM}"
        )
    for i, v in enumerate(vec):
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"{name} embedding[{i}] is {v} — contains NaN/Inf. "
                "Likely caused by a corrupt image or model precision issue."
            )


class VectorDBService:
    def __init__(self):
        # Read env vars at instantiation time (not import time) so the
        # offload_index command can override QDRANT_URL/API_KEY via CLI flags.
        import os

        url = os.getenv("QDRANT_URL", settings.QDRANT_URL)
        api_key = os.getenv("QDRANT_API_KEY", settings.QDRANT_API_KEY)
        self.client = QdrantClient(url=url, api_key=api_key)
        self.collection_name = os.getenv(
            "QDRANT_COLLECTION", settings.QDRANT_COLLECTION
        )

    def ensure_collection(self, vector_size: int = 1024):
        """
        Ensure the Qdrant collection exists with named 'image', 'text',
        and 'sparse' vectors.
        """
        # Verify connectivity first
        self.client.get_collections()
        collections = None
        try:
            collections = self.client.get_collections().collections
        except Exception as e:
            raise RuntimeError(
                f"Cannot reach Qdrant at {settings.QDRANT_URL}: {e}"
            ) from e
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    name: VectorParams(size=vector_size, distance=Distance.COSINE)
                    for name in VECTOR_NAMES
                },
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: SparseVectorParams(),
                },
            )

    def has_sparse_support(self) -> bool:
        """Check whether the current collection supports sparse vectors."""
        info = self.client.get_collection(collection_name=self.collection_name)
        return SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})

    def upsert_embedding(
        self,
        vector_id: uuid.UUID,
        image_embedding: list[float],
        text_embedding: list[float],
        payload: dict[str, Any],
        *,
        sparse_embedding: dict[str, list[int] | list[float]] | None = None,
    ):
        """
        Upsert image and text embeddings as named vectors for a single point.

        Optionally includes a sparse vector for lexical matching. If sparse
        embedding is provided but the collection doesn't support it, the
        upsert proceeds without the sparse component.
        """
        _validate_vector(image_embedding, "Image")
        _validate_vector(text_embedding, "Text")

        vectors: dict[str, list[float] | SparseVector] = {  # noqa: SIM908
            "image": image_embedding,
            "text": text_embedding,
        }
        if sparse_embedding is not None:
            vectors[SPARSE_VECTOR_NAME] = SparseVector(
                indices=sparse_embedding["indices"],  # ty: ignore
                values=sparse_embedding["values"],  # ty: ignore
            )

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(vector_id),
                    vector=vectors,  # ty: ignore
                    payload=payload,
                )
            ],
        )
