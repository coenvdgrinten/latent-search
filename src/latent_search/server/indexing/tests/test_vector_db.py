import uuid
from typing import override
from unittest.mock import MagicMock, patch

from django.test import TestCase
from latent_search.server.indexing.services.vector_db import VectorDBService


class VectorDBServiceTest(TestCase):
    @override
    def setUp(self):
        # Patch QdrantClient for all tests so no real connection is attempted
        patcher = patch("latent_search.server.indexing.services.vector_db.QdrantClient")
        self.mock_qdrant_class = patcher.start()
        self.mock_client = MagicMock()
        self.mock_qdrant_class.return_value = self.mock_client
        self.addCleanup(patcher.stop)

        self.service = VectorDBService()

    def test_ensure_collection_creates_when_not_exists(self):
        """Should call create_collection when the collection is absent."""
        self.mock_client.get_collections.return_value.collections = []

        self.service.ensure_collection(vector_size=1024)

        self.mock_client.create_collection.assert_called_once()
        kwargs = self.mock_client.create_collection.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], self.service.collection_name)
        config = kwargs["vectors_config"]
        self.assertIn("image", config)
        self.assertIn("text", config)

    def test_ensure_collection_skips_when_already_exists(self):
        """Should not call create_collection when the collection already exists."""
        existing = MagicMock()
        existing.name = self.service.collection_name
        self.mock_client.get_collections.return_value.collections = [existing]

        self.service.ensure_collection(vector_size=1024)

        self.mock_client.create_collection.assert_not_called()

    def test_upsert_embedding_passes_correct_args(self):
        """upsert_embedding should forward vector_id, both embeddings, and payload."""
        vector_id = uuid.uuid4()
        image_embedding = [0.1] * 1024
        text_embedding = [0.2] * 1024
        payload = {"file_path": "/tmp/photo.jpg"}

        self.service.upsert_embedding(
            vector_id=vector_id,
            image_embedding=image_embedding,
            text_embedding=text_embedding,
            payload=payload,
        )

        self.mock_client.upsert.assert_called_once()
        kwargs = self.mock_client.upsert.call_args.kwargs
        self.assertEqual(kwargs["collection_name"], self.service.collection_name)
        points = kwargs["points"]
        self.assertEqual(len(points), 1)
        self.assertEqual(points[0].id, str(vector_id))
        self.assertEqual(
            points[0].vector,
            {"image": image_embedding, "text": text_embedding},
        )
        self.assertEqual(points[0].payload, payload)
