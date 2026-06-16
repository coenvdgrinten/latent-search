"""Tests for ExifService — EXIF metadata extraction."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from latent_search.server.indexing.services.exif import (
    ExifService,
    MediaMetadata,
    _dms_to_decimal,
)


class DmsToDecimalTest(TestCase):
    """Unit tests for the _dms_to_decimal helper."""

    def test_north_hemisphere_positive(self):
        """North latitude should be positive."""
        result = _dms_to_decimal((51, 30, 0), "N")
        self.assertAlmostEqual(result, 51.5)

    def test_south_hemisphere_negative(self):
        """South latitude should be negative."""
        result = _dms_to_decimal((33, 51, 30), "S")
        self.assertAlmostEqual(result, -(33 + 51 / 60 + 30 / 3600))

    def test_east_longitude_positive(self):
        """East longitude should be positive."""
        result = _dms_to_decimal((0, 7, 30), "E")
        self.assertAlmostEqual(result, 0.125)

    def test_west_longitude_negative(self):
        """West longitude should be negative."""
        result = _dms_to_decimal((74, 0, 0), "W")
        self.assertAlmostEqual(result, -74.0)

    def test_zero_coordinates(self):
        """Null island (0, 0) should round-trip cleanly."""
        result = _dms_to_decimal((0, 0, 0), "N")
        self.assertEqual(result, 0.0)


class ExifServiceTest(TestCase):
    """Tests for ExifService.read_metadata."""

    @override
    def setUp(self):
        self.service = ExifService()
        self.tmpdir = TemporaryDirectory()

    @override
    def tearDown(self):
        self.tmpdir.cleanup()

    def _write_jpeg(self, filename: str, data: bytes) -> Path:
        """Write raw JPEG bytes to a temp file."""
        path = Path(self.tmpdir.name) / filename
        path.write_bytes(data)
        return path

    def test_returns_empty_metadata_on_no_exif(self):
        """Images without EXIF should return all-None metadata."""
        # Minimal valid JPEG (1x1 white pixel)
        jpeg_data = (
            b"\xff\xd8\xff\xe0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
            b"\xff\xd9"
        )
        path = self._write_jpeg("no_exif.jpg", jpeg_data)
        meta = self.service.read_metadata(path)

        # PIL may or may not read dimensions from a minimal JPEG stub
        # depending on version; the important thing is no exception
        self.assertIsInstance(meta, MediaMetadata)
        self.assertIsNone(meta.taken_at)
        self.assertIsNone(meta.latitude)
        self.assertIsNone(meta.longitude)

    def test_handles_missing_file_gracefully(self):
        """Missing files should return empty metadata, not raise."""
        meta = self.service.read_metadata("/nonexistent/path/photo.jpg")
        self.assertIsInstance(meta, MediaMetadata)
        self.assertIsNone(meta.taken_at)

    @patch("latent_search.server.indexing.services.exif.PilImage.open")
    def test_catches_pil_read_error(self, mock_open):
        """Corrupt images should log a warning, not crash."""
        mock_open.side_effect = OSError("cannot identify image file")
        meta = self.service.read_metadata("corrupt.jpg")

        self.assertIsInstance(meta, MediaMetadata)
        self.assertIsNone(meta.width)
        self.assertIsNone(meta.height)

    @patch("latent_search.server.indexing.services.exif.ExifImage")
    def test_catches_exif_read_error(self, mock_exif_cls):
        """EXIF read failures should fall back to empty metadata."""
        mock_exif_instance = MagicMock()
        mock_exif_instance.has_exif = False
        mock_exif_cls.return_value = mock_exif_instance

        meta = self.service.read_metadata("some.jpg")
        self.assertIsInstance(meta, MediaMetadata)

    @patch("latent_search.server.indexing.services.exif.ExifImage")
    @patch("latent_search.server.indexing.services.exif.PilImage.open")
    def test_extract_dimensions_from_pil(self, mock_pil_open, mock_exif_cls):
        """Width and height should come from PIL."""
        mock_img = MagicMock()
        mock_img.size = (1920, 1080)
        mock_pil_open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_pil_open.return_value.__exit__ = MagicMock(return_value=False)

        mock_exif_instance = MagicMock()
        mock_exif_instance.has_exif = False
        mock_exif_cls.return_value = mock_exif_instance

        meta = self.service.read_metadata("test.jpg")
        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.height, 1080)

    @patch("latent_search.server.indexing.services.exif.ExifImage")
    @patch("latent_search.server.indexing.services.exif.PilImage.open")
    def test_skips_gps_when_no_exif(self, mock_pil_open, mock_exif_cls):
        """GPS coordinates should be None when no EXIF present."""
        mock_img = MagicMock()
        mock_img.size = (800, 600)
        mock_pil_open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_pil_open.return_value.__exit__ = MagicMock(return_value=False)

        mock_exif_instance = MagicMock()
        mock_exif_instance.has_exif = False
        mock_exif_cls.return_value = mock_exif_instance

        meta = self.service.read_metadata("test.jpg")
        self.assertIsNone(meta.latitude)
        self.assertIsNone(meta.longitude)

    @patch("latent_search.server.indexing.services.exif.ExifImage")
    @patch("latent_search.server.indexing.services.exif.PilImage.open")
    def test_catches_gps_parse_error(self, mock_pil_open, mock_exif_cls):
        """Malformed GPS data should not propagate exceptions."""
        mock_img = MagicMock()
        mock_img.size = (100, 100)
        mock_pil_open.return_value.__enter__ = MagicMock(return_value=mock_img)
        mock_pil_open.return_value.__exit__ = MagicMock(return_value=False)

        mock_exif_instance = MagicMock()
        mock_exif_instance.has_exif = True
        mock_exif_instance.datetime_original = None
        # Simulate GPS attribute error
        del mock_exif_instance.gps_latitude

        meta = self.service.read_metadata("test.jpg")
        self.assertIsInstance(meta, MediaMetadata)
        self.assertIsNone(meta.latitude)
        self.assertIsNone(meta.longitude)


class MediaMetadataTest(TestCase):
    """Tests for the MediaMetadata dataclass."""

    def test_defaults_are_none(self):
        """All fields should default to None."""
        meta = MediaMetadata()
        self.assertIsNone(meta.width)
        self.assertIsNone(meta.height)
        self.assertIsNone(meta.taken_at)
        self.assertIsNone(meta.latitude)
        self.assertIsNone(meta.longitude)

    def test_accepts_partial_data(self):
        """Should work with only some fields populated."""
        meta = MediaMetadata(width=1920, height=1080)
        self.assertEqual(meta.width, 1920)
        self.assertEqual(meta.height, 1080)
        self.assertIsNone(meta.taken_at)
