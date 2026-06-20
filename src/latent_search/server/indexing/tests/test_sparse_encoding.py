"""Tests for the sparse encoding service."""

from unittest.mock import MagicMock, patch

import torch
from django.test import TestCase

from latent_search.server.indexing.services.sparse_encoding import (
    DEFAULT_SPARSE_MODEL_ID,
    SparseEncodingService,
    _tensor_to_qdrant_sparse,
)

_MODEL_PATCH = "latent_search.server.indexing.services.sparse_encoding.SparseEncoder"


class DefaultModelIdTest(TestCase):
    def test_default_model_id_is_splade_distilbert(self):
        self.assertIn("splade", DEFAULT_SPARSE_MODEL_ID, msg="Should use SPLADE")


class TensorConversionTest(TestCase):
    """Tests for the internal tensor-to-sparse converter."""

    def test_converts_1d_tensor_correctly(self):
        tensor = torch.tensor([0.0, 0.5, 0.0, 0.3])
        result = _tensor_to_qdrant_sparse(tensor)

        self.assertEqual(result["indices"], [1, 3])
        self.assertAlmostEqual(result["values"][0], 0.5)
        self.assertAlmostEqual(result["values"][1], 0.3)

    def test_filters_out_zeros(self):
        tensor = torch.tensor([0.0, 0.0, 0.0])
        result = _tensor_to_qdrant_sparse(tensor)

        self.assertEqual(len(result["indices"]), 0)
        self.assertEqual(len(result["values"]), 0)

    def test_handles_single_row_batch(self):
        tensor = torch.tensor([[0.1, 0.0, 0.2]])
        result = _tensor_to_qdrant_sparse(tensor)

        self.assertEqual(result["indices"], [0, 2])

    def test_returns_lists_not_tensors(self):
        tensor = torch.tensor([0.5, 0.3])
        result = _tensor_to_qdrant_sparse(tensor)

        assert isinstance(result["indices"], list)
        assert isinstance(result["values"], list)

    def test_handles_sparse_coo_tensor(self):
        """SPLADE returns COO-encoded sparse tensors — must handle gracefully."""
        # Create a sparse tensor matching SPLADE output shape (vocab size ~30k+)
        vocab_size = 100
        indices = torch.tensor([[2, 15, 42]])
        values = torch.tensor([0.8, 0.6, 0.3])
        sparse_tensor = torch.sparse_coo_tensor(
            indices, values, (vocab_size,), dtype=torch.float32
        )
        result = _tensor_to_qdrant_sparse(sparse_tensor)

        self.assertEqual(result["indices"], [2, 15, 42])
        self.assertAlmostEqual(result["values"][0], 0.8)
        self.assertAlmostEqual(result["values"][1], 0.6)
        self.assertAlmostEqual(result["values"][2], 0.3)


class SparseEncodingServiceInitTest(TestCase):
    """Tests for SparseEncodingService initialization."""

    def test_default_model_id(self):
        service = SparseEncodingService()
        self.assertEqual(service.model_id, DEFAULT_SPARSE_MODEL_ID)

    def test_custom_model_id(self):
        service = SparseEncodingService(model_id="custom/model")
        self.assertEqual(service.model_id, "custom/model")

    def test_model_starts_as_none(self):
        service = SparseEncodingService()
        self.assertIsNone(service._model)


class SparseEncodingServiceEncodeDocumentTest(TestCase):
    """Tests for the encode_document method."""

    @patch(_MODEL_PATCH)
    def test_encode_calls_model_with_text(self, mock_sp_class):
        service = SparseEncodingService()
        mock_model = MagicMock()
        mock_model.encode_document.return_value = torch.tensor([0.1, 0.2, 0.0])
        mock_sp_class.return_value = mock_model

        service.encode_document("hello world")

        mock_model.encode_document.assert_called_once_with("hello world")

    @patch(_MODEL_PATCH)
    def test_encode_returns_qdrant_sparse_format(self, mock_sp_class):
        service = SparseEncodingService()
        mock_model = MagicMock()
        mock_model.encode_document.return_value = torch.tensor([0.0, 0.5, 0.3])
        mock_sp_class.return_value = mock_model

        result = service.encode_document("test caption")

        assert isinstance(result, dict)
        assert "indices" in result
        assert "values" in result
        self.assertEqual(result["indices"], [1, 2])


class SparseEncodingServiceBatchEncodeTest(TestCase):
    """Tests for batch encoding."""

    @patch(_MODEL_PATCH)
    def test_empty_input_returns_empty_list(self, mock_sp_class):
        service = SparseEncodingService()
        results = service.batch_encode_documents([])
        self.assertEqual(results, [])

    @patch(_MODEL_PATCH)
    def test_batch_encodes_multiple_texts(self, mock_sp_class):
        service = SparseEncodingService()
        mock_model = MagicMock()
        mock_model.encode_document.return_value = torch.tensor([[0.1, 0.0], [0.0, 0.2]])
        mock_sp_class.return_value = mock_model

        results = service.batch_encode_documents(["doc one", "doc two"])

        self.assertEqual(len(results), 2)
        assert all(isinstance(r, dict) for r in results)
