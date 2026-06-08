import math
import uuid
from typing import Any

from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

VECTOR_NAMES = ("image", "text")
EXPECTED_DIM = 1024


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
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int = 1024):
        """
        Ensure the Qdrant collection exists with named 'image' and 'text' vectors.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config={
                    name: VectorParams(size=vector_size, distance=Distance.COSINE)
                    for name in VECTOR_NAMES
                },
            )

    def upsert_embedding(
        self,
        vector_id: uuid.UUID,
        image_embedding: list[float],
        text_embedding: list[float],
        payload: dict[str, Any],
    ):
        """
        Upsert image and text embeddings as named vectors for a single point.
        """
        _validate_vector(image_embedding, "Image")
        _validate_vector(text_embedding, "Text")

        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(vector_id),
                    vector={"image": image_embedding, "text": text_embedding},
                    payload=payload,
                )
            ],
        )
