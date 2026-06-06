from typing import override
from unittest.mock import MagicMock, patch

import torch
from django.test import TestCase

from ..services.clip import CLIPService


class CLIPServiceTest(TestCase):
    @override
    def setUp(self):
        self.service = CLIPService()

    @patch("latent_search.server.indexing.services.clip.CLIPModel.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.CLIPProcessor.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.Image.open")
    def test_get_image_embedding(
        self, mock_image_open, mock_processor_init, mock_model_init
    ):
        mock_processor = MagicMock()
        mock_processor_init.return_value = mock_processor

        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model_init.return_value = mock_model

        mock_features = torch.randn(1, 512)
        mock_model.get_image_features.return_value = mock_features
        mock_model.to.return_value = mock_model  # Handle .to(device)

        with patch("torch.no_grad"):
            embedding = self.service.get_image_embedding("fake_path.jpg")

        self.assertEqual(len(embedding), 512)
        mock_image_open.assert_called_once_with("fake_path.jpg")
        mock_model.get_image_features.assert_called_once()

    @patch("latent_search.server.indexing.services.clip.CLIPModel.from_pretrained")
    @patch("latent_search.server.indexing.services.clip.CLIPProcessor.from_pretrained")
    def test_get_text_embedding(self, mock_processor_init, mock_model_init):
        mock_processor = MagicMock()
        mock_processor_init.return_value = mock_processor

        mock_inputs = MagicMock()
        mock_processor.return_value = mock_inputs
        mock_inputs.to.return_value = mock_inputs

        mock_model = MagicMock()
        mock_model_init.return_value = mock_model

        mock_features = torch.randn(1, 512)
        mock_model.get_text_features.return_value = mock_features
        mock_model.to.return_value = mock_model

        embedding = self.service.get_text_embedding("a photo of a cat")

        self.assertEqual(len(embedding), 512)
        mock_model.get_text_features.assert_called_once()
