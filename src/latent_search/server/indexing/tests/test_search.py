from typing import override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from latent_search.server.indexing.services.search import SearchService


class SearchServiceTest(TestCase):
    @override
    def setUp(self):
        text_emb_patcher = patch(
            "latent_search.server.indexing.services.search.TextEmbeddingService"
        )
        sparse_patcher = patch(
            "latent_search.server.indexing.services.search.SparseEncodingService"
        )
        qdrant_patcher = patch(
            "latent_search.server.indexing.services.search.QdrantClient"
        )
        self.mock_text_emb_class = text_emb_patcher.start()
        self.mock_sparse_class = sparse_patcher.start()
        self.mock_qdrant_class = qdrant_patcher.start()
        self.addCleanup(text_emb_patcher.stop)
        self.addCleanup(sparse_patcher.stop)
        self.addCleanup(qdrant_patcher.stop)

        self.mock_text_emb = self.mock_text_emb_class.return_value
        self.mock_sparse = self.mock_sparse_class.return_value
        self.mock_client = self.mock_qdrant_class.return_value
        self.mock_text_emb.encode.return_value = [0.1] * 1024
        # Disable sparse support in tests — collection has no sparse config.
        self.mock_client.get_collection.return_value.config.params.sparse_vectors = {}

        self.service = SearchService()

    def _mock_query_points(self, hits: list) -> MagicMock:
        """Helper to set up mock_client.query_points to return a list of hits."""
        mock_result = MagicMock()
        mock_result.points = hits
        self.mock_client.query_points.return_value = mock_result
        return mock_result

    def test_semantic_search_returns_correctly_shaped_dicts(self):
        """Each result dict should contain id, score, file_path, and file_name."""
        mock_hit = MagicMock()
        mock_hit.id = "abc-123"
        mock_hit.score = 0.95
        mock_hit.payload = {"file_path": "/tmp/photo.jpg", "file_name": "photo.jpg"}
        self._mock_query_points([mock_hit])

        results = self.service.semantic_search("a cat on a beach", limit=5)

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["id"], "abc-123")
        self.assertEqual(results[0]["score"], 0.95)
        self.assertEqual(results[0]["file_path"], "/tmp/photo.jpg")
        self.assertEqual(results[0]["file_name"], "photo.jpg")

    def test_semantic_search_passes_query_embedding_to_qdrant(self):
        """Query embedding should reach Qdrant via prefetch + RRF."""
        self._mock_query_points([])

        self.service.semantic_search("sunset over the ocean", limit=10)

        self.mock_text_emb.encode.assert_called_once_with("sunset over the ocean")
        search_kwargs = self.mock_client.query_points.call_args.kwargs
        # Query is now an RRF fusion object (dual-vector when sparse unsupported).
        from qdrant_client.models import Rrf, RrfQuery

        self.assertIsInstance(search_kwargs["query"], RrfQuery)
        self.assertIsInstance(search_kwargs["query"].rrf, Rrf)
        self.assertEqual(search_kwargs["limit"], 10)
        self.assertIn("prefetch", search_kwargs)
        # Two prefetches: image-vector and text-vector.
        self.assertEqual(len(search_kwargs["prefetch"]), 2)
        usings = [p.using for p in search_kwargs["prefetch"]]
        self.assertIn("image", usings)
        self.assertIn("text", usings)

    def test_semantic_search_returns_empty_list_when_no_results(self):
        """Should return an empty list when Qdrant returns no hits."""
        self._mock_query_points([])

        results = self.service.semantic_search("nothing matches")

        self.assertEqual(results, [])

    def test_semantic_search_handles_missing_payload_keys(self):
        """Results with partial payloads should default missing keys to empty string."""
        mock_hit = MagicMock()
        mock_hit.id = "xyz"
        mock_hit.score = 0.5
        mock_hit.payload = {}
        self._mock_query_points([mock_hit])

        results = self.service.semantic_search("test query")

        self.assertEqual(results[0]["file_path"], "")
        self.assertEqual(results[0]["file_name"], "")
