from typing import override
from unittest.mock import MagicMock, patch

import numpy as np
from django.test import TestCase

from ..services.clip import CLIPService


class CLIPServiceTest(TestCase):
    @override
    def setUp(self):
        self.service = CLIPService()
        # Simulate a batch of 1 embedding at 1024 dimensions
        self.mock_embeddings = np.zeros((1, 1024), dtype=np.float32)

    def _make_mock_model(self, mock_automodel_init: MagicMock) -> MagicMock:
        mock_model = MagicMock()
        mock_automodel_init.return_value = mock_model
        mock_model.encode_image.return_value = self.mock_embeddings
        mock_model.encode_text.return_value = self.mock_embeddings
        return mock_model

    @patch("latent_search.server.indexing.services.clip.AutoModel.from_pretrained")
    def test_get_image_embedding(self, mock_automodel_init):
        mock_model = self._make_mock_model(mock_automodel_init)

        embedding = self.service.get_image_embedding("fake_path.jpg")

        self.assertEqual(len(embedding), 1024)
        mock_model.encode_image.assert_called_once_with(
            ["fake_path.jpg"], truncate_dim=1024
        )

    @patch("latent_search.server.indexing.services.clip.AutoModel.from_pretrained")
    def test_get_image_embedding_error(self, mock_automodel_init):
        mock_model = self._make_mock_model(mock_automodel_init)
        mock_model.encode_image.side_effect = FileNotFoundError("File not found")

        with self.assertRaises(FileNotFoundError):
            self.service.get_image_embedding("non_existent.jpg")

    @patch("latent_search.server.indexing.services.clip.AutoModel.from_pretrained")
    def test_get_text_embedding(self, mock_automodel_init):
        mock_model = self._make_mock_model(mock_automodel_init)

        embedding = self.service.get_text_embedding("a photo of a cat")

        self.assertEqual(len(embedding), 1024)
        mock_model.encode_text.assert_called_once_with(
            ["a photo of a cat"], truncate_dim=1024, task="retrieval.query"
        )

    @patch("latent_search.server.indexing.services.clip.AutoModel.from_pretrained")
    def test_get_text_embedding_passage_task(self, mock_automodel_init):
        mock_model = self._make_mock_model(mock_automodel_init)

        self.service.get_text_embedding("london bridge", task=None)

        mock_model.encode_text.assert_called_once_with(
            ["london bridge"], truncate_dim=1024, task=None
        )
