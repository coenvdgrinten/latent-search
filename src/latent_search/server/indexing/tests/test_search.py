from typing import override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from latent_search.server.indexing.services.search import SearchService


class SearchServiceTest(TestCase):
    @override
    def setUp(self):
        clip_patcher = patch(
            "latent_search.server.indexing.services.search.CLIPService"
        )
        qdrant_patcher = patch(
            "latent_search.server.indexing.services.search.QdrantClient"
        )
        self.mock_clip_class = clip_patcher.start()
        self.mock_qdrant_class = qdrant_patcher.start()
        self.addCleanup(clip_patcher.stop)
        self.addCleanup(qdrant_patcher.stop)

        self.mock_clip = self.mock_clip_class.return_value
        self.mock_client = self.mock_qdrant_class.return_value
        self.mock_clip.get_text_embedding.return_value = [0.1] * 512

        self.service = SearchService()

    def test_semantic_search_returns_correctly_shaped_dicts(self):
        """Each result dict should contain id, score, file_path, and file_name."""
        mock_hit = MagicMock()
        mock_hit.id = "abc-123"
        mock_hit.score = 0.95
        mock_hit.payload = {"file_path": "/tmp/photo.jpg", "file_name": "photo.jpg"}
        self.mock_client.search.return_value = [mock_hit]

        results = self.service.semantic_search("a cat on a beach", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "abc-123")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["file_path"], "/tmp/photo.jpg")
        self.assertEqual(results[0]["file_name"], "photo.jpg")

    def test_semantic_search_passes_query_embedding_to_qdrant(self):
        """The text embedding for the query should be forwarded to Qdrant."""
        self.mock_client.search.return_value = []

        self.service.semantic_search("sunset over the ocean", limit=10)

        self.mock_clip.get_text_embedding.assert_called_once_with(
            "sunset over the ocean"
        )
        search_kwargs = self.mock_client.search.call_args.kwargs
        self.assertEqual(search_kwargs["query_vector"], [0.1] * 512)
        self.assertEqual(search_kwargs["limit"], 10)

    def test_semantic_search_returns_empty_list_when_no_results(self):
        """Should return an empty list when Qdrant returns no hits."""
        self.mock_client.search.return_value = []

        results = self.service.semantic_search("nothing matches")

        self.assertEqual(results, [])

    def test_semantic_search_handles_missing_payload_keys(self):
        """Results with partial payloads should default missing keys to empty string."""
        mock_hit = MagicMock()
        mock_hit.id = "xyz"
        mock_hit.score = 0.5
        mock_hit.payload = {}
        self.mock_client.search.return_value = [mock_hit]

        results = self.service.semantic_search("test query")

        self.assertEqual(results[0]["file_path"], "")
        self.assertEqual(results[0]["file_name"], "")
