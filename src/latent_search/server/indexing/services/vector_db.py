import uuid
from typing import Any

import numpy as np
from django.conf import settings
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


class VectorDBService:
    def __init__(self):
        self.client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def ensure_collection(self, vector_size: int = 512):
        """
        Ensure the Qdrant collection exists with correct parameters.
        """
        collections = self.client.get_collections().collections
        exists = any(c.name == self.collection_name for c in collections)

        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def upsert_embedding(
        self, vector_id: uuid.UUID, embedding: list[float], payload: dict[str, Any]
    ):
        """
        Upsert a single embedding to Qdrant.
        """
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=str(vector_id),
                    vector=embedding,
                    payload=payload,
                )
            ],
        )
