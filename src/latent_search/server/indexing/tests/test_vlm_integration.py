"""Integration tests for VLMService against real image files.

These tests run the actual Qwen2.5-VL-3B model on test images stored
in the media/ directory. They are slow (several minutes per image on CPU)
and will skip if no test images are found.

Run with:
    ./manage test latent_search.server.indexing.tests.test_vlm_integration
"""

from pathlib import Path
from typing import override

from django.conf import settings
from django.test import TestCase

from latent_search.server.indexing.services.vlm import VLMService


class VLMServiceIntegrationTest(TestCase):
    """Real-model tests for VLM caption generation."""

    @classmethod
    def find_test_images(cls) -> list[Path]:
        """Locate test images in the project's media/ directory."""
        media_root = Path(settings.MEDIA_ROOT).resolve()
        if not media_root.is_dir():
            return []
        return sorted(media_root.glob("*.jpg"))

    @classmethod
    @override
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.images = cls.find_test_images()
        cls._has_images = bool(cls.images)

    def test_describe_returns_non_empty_caption(self):
        """describe() should produce a non-empty string for a valid image."""
        if not self._has_images:
            self.skipTest("No test images found in media/")
        service = VLMService()
        caption = service.describe(self.images[0])

        self.assertIsInstance(caption, str)
        self.assertTrue(
            len(caption) > 0,
            f"Caption is empty for {self.images[0].name}",
        )
        # VLM system prompt caps at ~200 chars
        self.assertLessEqual(
            len(caption),
            300,
            (
                f"Caption seems excessively long "
                f"({len(caption)} chars) for {self.images[0].name}"
            ),
        )

    def test_describe_different_images_produce_different_captions(self):
        """Different images should produce meaningfully different captions."""
        if not self._has_images:
            self.skipTest("No test images found in media/")
        if len(self.images) < 2:
            self.skipTest("Need at least 2 test images")

        service = VLMService()
        captions = [service.describe(img) for img in self.images[:2]]

        self.assertNotEqual(
            captions[0],
            captions[1],
            (
                f"Captions are identical for {self.images[0].name} "
                f"and {self.images[1].name}: {captions[0]}"
            ),
        )

    def test_describe_batch_handles_multiple_images(self):
        """describe_batch() should return one caption per input image."""
        if not self._has_images:
            self.skipTest("No test images found in media/")
        service = VLMService()
        paths = [str(img) for img in self.images[:2]]
        captions = service.describe_batch(paths)

        self.assertEqual(len(captions), len(paths))
        for cap, _path in zip(captions, paths, strict=True):
            self.assertIsInstance(cap, str)

    def test_describe_batch_skips_missing_file_gracefully(self):
        """describe_batch() should return empty string for missing files."""
        service = VLMService()
        paths = ["/nonexistent/image.jpg"]
        captions = service.describe_batch(paths)

        self.assertEqual(len(captions), 1)
        self.assertEqual(captions[0], "")
