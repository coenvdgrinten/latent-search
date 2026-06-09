"""Tests for the VLM caption enrichment service."""

import uuid
from datetime import datetime
from unittest.mock import patch

from django.test import TestCase

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.indexing import IndexingService

_CLIP_PATCH = "latent_search.server.indexing.services.indexing.CLIPService"
_TEXT_EMBED_PATCH = (
    "latent_search.server.indexing.services.indexing.TextEmbeddingService"
)
_VDB_PATCH = "latent_search.server.indexing.services.indexing.VectorDBService"
_DISCOVERY_PATCH = "latent_search.server.indexing.services.indexing.DiscoveryService"
_GEO_PATCH = "latent_search.server.indexing.services.indexing.GeocodingService"
_VLM_PATCH = "latent_search.server.indexing.services.indexing.VLMService"


class IndexingServiceVLMTest(TestCase):
    """Test VLM caption integration in the indexing pipeline."""

    @patch(_VLM_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_DISCOVERY_PATCH)
    def test_index_uses_vlm_caption_when_available(
        self,
        mock_discovery_class,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_vlm_class,
    ):  # noqa: E501
        """When a media has a vlm_caption, it should be prepended."""
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        _ = mock_vdb_class  # Used by decorator

        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/test.jpg",
            filename="test.jpg",
            relative_path="test.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False,
            vlm_caption=(
                "A beautiful sunset over the ocean "
                "with palm trees silhouetted against orange sky"
            ),
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertTrue(media.is_indexed)

        # Verify the caption includes the VLM description
        self.assertIn("beautiful sunset", media.caption)
        self.assertIn("ocean", media.caption)

    @patch(_VLM_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_DISCOVERY_PATCH)
    def test_index_without_vlm_caption(
        self,
        mock_discovery_class,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_vlm_class,
    ):  # noqa: E501
        """When no vlm_caption exists, indexing should work normally."""
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        _ = mock_vdb_class  # Used by decorator

        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/beach_photo.jpg",
            filename="beach_photo.jpg",
            relative_path="beach_photo.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False,
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertTrue(media.is_indexed)
        # Caption should contain filename keywords but no VLM content
        self.assertIn("beach", media.caption)
        self.assertNotIn("A beautiful", media.caption)

    @patch(_VLM_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_DISCOVERY_PATCH)
    def test_vlm_caption_order_in_combined_caption(
        self,
        mock_discovery_class,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_vlm_class,
    ):  # noqa: E501
        """VLM caption should come first, followed by factual data."""
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        _ = mock_vdb_class  # Used by decorator
        mock_geo = mock_geo_class.return_value
        mock_geo.reverse_geocode.return_value = "Paris, France"

        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/eiffel_tower.jpg",
            filename="eiffel_tower.jpg",
            relative_path="paris/eiffel_tower.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False,
            taken_at=datetime(2023, 7, 15, 18, 30, 0),
            latitude=48.8566,
            longitude=2.3522,
            vlm_caption=(
                "The Eiffel Tower illuminated at dusk "
                "with lights twinkling on the iron lattice structure"
            ),
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        caption = media.caption

        # VLM caption should be first
        self.assertTrue(caption.startswith("The Eiffel Tower illuminated"))
        # Factual data should follow
        self.assertIn("eiffel", caption)
        self.assertIn("paris", caption)
        self.assertIn("france", caption)

    @patch(_VLM_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_DISCOVERY_PATCH)
    def test_vlm_service_initialized_on_indexing(
        self,
        mock_discovery_class,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_vlm_class,
    ):
        """VLM service should be initialized in IndexingService constructor."""
        service = IndexingService()
        self.assertIsNotNone(service.vlm)
        mock_vlm_class.assert_called_once()
