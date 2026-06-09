import tempfile
from pathlib import Path
from typing import override

from django.test import TestCase

from latent_search.server.indexing.services.discovery import DiscoveryService


class DiscoveryServiceTest(TestCase):
    @override
    def setUp(self):
        self.service = DiscoveryService()

    def test_discovers_supported_extensions(self):
        """Only jpg, jpeg, png, and webp files should be returned."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_bytes(b"")
            (root / "image.jpeg").write_bytes(b"")
            (root / "picture.png").write_bytes(b"")
            (root / "web.webp").write_bytes(b"")
            (root / "document.pdf").write_bytes(b"")
            (root / "notes.txt").write_bytes(b"")

            results = list(self.service.discover_media(root))
            names = {p.name for p in results}

            self.assertIn("photo.jpg", names)
            self.assertIn("image.jpeg", names)
            self.assertIn("picture.png", names)
            self.assertIn("web.webp", names)
            self.assertNotIn("document.pdf", names)
            self.assertNotIn("notes.txt", names)

    def test_extension_matching_is_case_insensitive(self):
        """Uppercase extensions like .JPG should be discovered."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "UPPER.JPG").write_bytes(b"")
            (root / "mixed.Png").write_bytes(b"")

            results = list(self.service.discover_media(root))
            names = {p.name for p in results}

            self.assertIn("UPPER.JPG", names)
            self.assertIn("mixed.Png", names)

    def test_raises_for_non_directory(self):
        """Passing a file path should raise NotADirectoryError."""
        with tempfile.NamedTemporaryFile() as tmp_file:
            with self.assertRaises(NotADirectoryError):
                list(self.service.discover_media(tmp_file.name))

    def test_recursive_discovery(self):
        """Files in subdirectories should be found."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subdir = root / "subdir" / "nested"
            subdir.mkdir(parents=True)
            (root / "top.jpg").write_bytes(b"")
            (subdir / "deep.png").write_bytes(b"")

            results = list(self.service.discover_media(root))
            names = {p.name for p in results}

            self.assertIn("top.jpg", names)
            self.assertIn("deep.png", names)
            self.assertEqual(len(results), 2)

    def test_empty_directory_returns_no_results(self):
        """An empty directory should yield an empty iterator."""
        with tempfile.TemporaryDirectory() as tmp:
            results = list(self.service.discover_media(tmp))
            self.assertEqual(results, [])
