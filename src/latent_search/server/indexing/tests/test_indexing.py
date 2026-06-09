import uuid
from datetime import datetime
from pathlib import Path
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


class IndexingServiceTest(TestCase):
    @patch(_DISCOVERY_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    def test_run_discovery(self, mock_vdb_class, mock_clip_class, mock_discovery_class):
        service = IndexingService()
        mock_discovery = mock_discovery_class.return_value
        mock_discovery.discover_media.return_value = [
            Path("/tmp/image1.jpg"),
            Path("/tmp/image2.png"),
        ]

        service.run_discovery("/tmp")

        self.assertEqual(IndexedMedia.objects.count(), 2)

    @patch(_DISCOVERY_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    def test_index_pending_media(
        self,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_discovery_class,
    ):
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        mock_vector_db = mock_vdb_class.return_value

        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/test.jpg",
            filename="test.jpg",
            relative_path="test.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False,
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertTrue(media.is_indexed)
        mock_clip.get_image_embedding.assert_called_once_with("/tmp/test.jpg")
        mock_vector_db.upsert_embedding.assert_called_once()

    @patch(_DISCOVERY_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    def test_run_discovery_deduplicates(
        self, mock_vdb_class, mock_clip_class, mock_discovery_class
    ):
        service = IndexingService()
        mock_discovery = mock_discovery_class.return_value
        mock_discovery.discover_media.return_value = [Path("/tmp/image1.jpg")]

        service.run_discovery("/tmp")
        service.run_discovery("/tmp")

        self.assertEqual(IndexedMedia.objects.count(), 1)

    @patch(_DISCOVERY_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    def test_index_pending_media_continues_on_error(
        self,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_discovery_class,
    ):
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_clip.get_image_embedding.side_effect = OSError("Corrupt file")

        media = IndexedMedia.objects.create(
            file_path="/tmp/bad.jpg",
            filename="bad.jpg",
            relative_path="bad.jpg",
            file_size=0,
            is_indexed=False,
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertFalse(media.is_indexed)

    @patch(_DISCOVERY_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    def test_index_pending_media_assigns_vector_id_when_missing(
        self,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_discovery_class,
    ):
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/no_id.jpg",
            filename="no_id.jpg",
            relative_path="no_id.jpg",
            file_size=512,
            vector_id=None,
            is_indexed=False,
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertTrue(media.is_indexed)
        self.assertIsNotNone(media.vector_id)

    @patch(_DISCOVERY_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    def test_caption_includes_location_and_temporal(
        self,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_discovery_class,
    ):
        """Caption should include reverse-geocoded location and temporal context."""
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024
        mock_geo = mock_geo_class.return_value
        mock_geo.reverse_geocode.return_value = "London, England, United Kingdom"

        IndexedMedia.objects.create(
            file_path="/home/user/photos/eiffel-tower.jpg",
            filename="eiffel-tower.jpg",
            relative_path="photos/eiffel-tower.jpg",
            file_size=2048,
            vector_id=uuid.uuid4(),
            is_indexed=False,
            latitude=48.8584,
            longitude=2.2945,
            taken_at=datetime(2022, 7, 15, 14, 30, 0),
        )

        service.index_pending_media(batch_size=1)

        call_args = mock_text_embed.encode.call_args
        caption = call_args.args[0]
        self.assertIn("eiffel tower", caption)
        self.assertIn("london", caption)
        self.assertIn("july", caption)
        self.assertIn("summer", caption)

    @patch(_DISCOVERY_PATCH)
    @patch(_TEXT_EMBED_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    def test_caption_gracefully_skips_missing_gps(
        self,
        mock_geo_class,
        mock_vdb_class,
        mock_clip_class,
        mock_text_embed_class,
        mock_discovery_class,
    ):
        """Missing GPS should not break caption generation."""
        service = IndexingService()
        mock_clip = mock_clip_class.return_value
        mock_text_embed = mock_text_embed_class.return_value
        mock_clip.get_image_embedding.return_value = [0.0] * 1024
        mock_text_embed.encode.return_value = [0.0] * 1024

        media = IndexedMedia.objects.create(
            file_path="/tmp/beach-day.jpg",
            filename="beach-day.jpg",
            relative_path="beach-day.jpg",
            file_size=1024,
            vector_id=uuid.uuid4(),
            is_indexed=False,
            latitude=None,
            longitude=None,
            taken_at=None,
        )

        service.index_pending_media(batch_size=1)

        media.refresh_from_db()
        self.assertTrue(media.is_indexed)
        mock_geo_class.return_value.reverse_geocode.assert_not_called()
