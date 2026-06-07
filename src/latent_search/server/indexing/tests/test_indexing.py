import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase
from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.indexing import IndexingService


class IndexingServiceTest(TestCase):
    @patch("latent_search.server.indexing.services.indexing.DiscoveryService")
    @patch("latent_search.server.indexing.services.indexing.CLIPService")
    @patch("latent_search.server.indexing.services.indexing.VectorDBService")
    def test_run_discovery(self, mock_vector_db_class, mock_clip_class, mock_discovery_class):
        service = IndexingService()
        # Setup
        mock_discovery = mock_discovery_class.return_value
        mock_discovery.discover_media.return_value = [
            Path("/tmp/image1.jpg"),
            Path("/tmp/image2.png"),
        ]

        # Execute
        service.run_discovery("/tmp")

        # Verify
        self.assertEqual(IndexedMedia.objects.count(), 2)

    @patch("latent_search.server.indexing.services.indexing.CLIPService")
    @patch("latent_search.server.indexing.services.indexing.VectorDBService")
    @patch("latent_search.server.indexing.services.indexing.DiscoveryService")
    def test_index_pending_media(self, mock_discovery_class, mock_vector_db_class, mock_clip_class):
        service = IndexingService()
        # Setup
        mock_clip = mock_clip_class.return_value
        mock_vector_db = mock_vector_db_class.return_value
        
        mock_clip.get_image_embedding.return_value = [0.0] * 512
        
        media = IndexedMedia.objects.create(
            file_path="/tmp/test.jpg",
            filename="test.jpg",
            relative_path="test.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False
        )

        # Execute
        service.index_pending_media(batch_size=1)

        # Verify
        media.refresh_from_db()
        self.assertTrue(media.is_indexed)
        mock_clip.get_image_embedding.assert_called_once_with("/tmp/test.jpg")
        mock_vector_db.upsert_embedding.assert_called_once()
