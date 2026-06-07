from typing import override
from unittest.mock import MagicMock, patch

import torch
from django.test import TestCase

from ..services.clip import CLIPService


class CLIPServiceTest(TestCase):
    @override
    def setUp(self):
        self.service = CLIPService()
        self.mock_features = torch.randn(1, 512)

    def _setup_mocks(self, mock_model_init, mock_processor_init):
        # Mock Processor
        mock_processor = MagicMock()
        mock_processor_init.return_value = mock_processor

        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        # Mock Model — wire up sub-models used by the v5 API
        mock_model = MagicMock()
        mock_model_init.return_value = mock_model
        mock_model.to.return_value = mock_model

        mock_vision_output = MagicMock()
        mock_vision_output.pooler_output = self.mock_features
        mock_model.vision_model.return_value = mock_vision_output
        mock_model.visual_projection.return_value = self.mock_features

        mock_text_output = MagicMock()
        mock_text_output.pooler_output = self.mock_features
        mock_model.text_model.return_value = mock_text_output
        mock_model.text_projection.return_value = self.mock_features

        return mock_model, mock_processor

    @patch("latent_search.server.indexing.services.clip.CLIPModel.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.CLIPProcessor.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.Image.open")
    def test_get_image_embedding(
        self, mock_image_open, mock_processor_init, mock_model_init
    ):
        mock_model, _ = self._setup_mocks(mock_model_init, mock_processor_init)

        # Execute
        embedding = self.service.get_image_embedding("fake_path.jpg")

        # Verify
        self.assertEqual(len(embedding), 512)
        mock_image_open.assert_called_once_with("fake_path.jpg")
        mock_model.vision_model.assert_called_once()

    @patch("latent_search.server.indexing.services.clip.CLIPModel.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.CLIPProcessor.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.Image.open")
    def test_get_image_embedding_error(
        self, mock_image_open, mock_processor_init, mock_model_init
    ):
        self._setup_mocks(mock_model_init, mock_processor_init)
        mock_image_open.side_effect = FileNotFoundError("File not found")

        # Execute and Verify
        with self.assertRaises(FileNotFoundError):
            self.service.get_image_embedding("non_existent.jpg")

    @patch("latent_search.server.indexing.services.clip.CLIPModel.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.CLIPProcessor.from_pretrained")
    def test_get_text_embedding(self, mock_processor_init, mock_model_init):
        mock_model, _ = self._setup_mocks(mock_model_init, mock_processor_init)

        # Execute
        embedding = self.service.get_text_embedding("a photo of a cat")

        # Verify
        self.assertEqual(len(embedding), 512)
        mock_model.text_model.assert_called_once()
