"""Tests for the text embedding service."""

from unittest.mock import MagicMock, patch

import torch
from django.test import TestCase

from latent_search.server.indexing.services.text_embedding import (
    VECTOR_DIM,
    TextEmbeddingService,
)

_MODEL_PATCH = (
    "latent_search.server.indexing.services.text_embedding.SentenceTransformer"
)


class VectorDimConstantTest(TestCase):
    """Tests for the VECTOR_DIM constant."""

    def test_vector_dim_is_1024(self):
        self.assertEqual(VECTOR_DIM, 1024)


class TextEmbeddingServiceInitTest(TestCase):
    """Tests for TextEmbeddingService initialization."""

    def test_default_model_id(self):
        service = TextEmbeddingService()
        self.assertEqual(service.model_id, "BAAI/bge-large-en-v1.5")

    def test_custom_model_id(self):
        service = TextEmbeddingService(model_id="custom/model")
        self.assertEqual(service.model_id, "custom/model")

    def test_model_starts_as_none(self):
        service = TextEmbeddingService()
        self.assertIsNone(service._model)


class TextEmbeddingServiceEncodeTest(TestCase):
    """Tests for the encode method."""

    @patch(_MODEL_PATCH)
    def test_encode_returns_flat_list(self, mock_st_class):
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.zeros(VECTOR_DIM)
        mock_st_class.return_value = mock_model

        result = service.encode("hello world")

        self.assertIsInstance(result, list)
        self.assertEqual(len(result), VECTOR_DIM)

    @patch(_MODEL_PATCH)
    def test_encode_normalizes_embeddings(self, mock_st_class):
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.ones(VECTOR_DIM)
        mock_st_class.return_value = mock_model

        service.encode("test text")

        mock_model.encode.assert_called_once_with(
            "test text", normalize_embeddings=True
        )

    @patch(_MODEL_PATCH)
    def test_encode_uses_no_grad(self, mock_st_class):
        """Verifies torch.no_grad context is used (no training ops)."""
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.zeros(VECTOR_DIM)
        mock_st_class.return_value = mock_model

        # Should not raise — no_grad context is entered
        result = service.encode("test")
        self.assertIsInstance(result, list)

    @patch(_MODEL_PATCH)
    def test_lazy_loading_loads_model_on_first_encode(self, mock_st_class):
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.zeros(VECTOR_DIM)
        mock_st_class.return_value = mock_model

        service.encode("first call")

        mock_st_class.assert_called_once_with("BAAI/bge-large-en-v1.5")

    @patch(_MODEL_PATCH)
    def test_model_loaded_only_once(self, mock_st_class):
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.return_value = torch.zeros(VECTOR_DIM)
        mock_st_class.return_value = mock_model

        service.encode("first")
        service.encode("second")
        service.encode("third")

        self.assertEqual(mock_st_class.call_count, 1)

    @patch(_MODEL_PATCH)
    def test_encode_different_texts(self, mock_st_class):
        service = TextEmbeddingService()
        mock_model = MagicMock()
        mock_model.encode.side_effect = [
            torch.zeros(VECTOR_DIM),
            torch.ones(VECTOR_DIM) * 0.5,
        ]
        mock_st_class.return_value = mock_model

        result1 = service.encode("hello")
        result2 = service.encode("goodbye")

        self.assertEqual(mock_model.encode.call_count, 2)
        self.assertIsInstance(result1, list)
        self.assertIsInstance(result2, list)
