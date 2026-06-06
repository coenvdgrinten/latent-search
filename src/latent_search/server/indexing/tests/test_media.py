import uuid
from typing import override

from django.test import TestCase
from django.utils import timezone

from ..models import IndexedMedia


class IndexedMediaModelTest(TestCase):
    @override
    def setUp(self):
        self.media = IndexedMedia.objects.create(
            file_path="/mnt/user/photos/vacation/beach.jpg",
            filename="beach.jpg",
            relative_path="vacation/beach.jpg",
            file_size=2048576,
            mime_type="image/jpeg",
        )

    def test_media_creation(self):
        """Test that media is correctly created with defaults."""
        self.assertEqual(self.media.filename, "beach.jpg")
        self.assertFalse(self.media.is_indexed)
        self.assertIsNone(self.media.vector_id)

    def test_string_representation(self):
        """Test the __str__ method."""
        self.assertEqual(str(self.media), "beach.jpg (Pending)")

        self.media.is_indexed = True
        self.media.save()
        self.assertEqual(str(self.media), "beach.jpg (Indexed)")

    def test_unique_file_path(self):
        """Test that duplicate file paths are prevented."""
        from django.db import IntegrityError

        with self.assertRaises(IntegrityError):
            IndexedMedia.objects.create(
                file_path="/mnt/user/photos/vacation/beach.jpg",
                filename="beach_copy.jpg",
                relative_path="vacation/beach_copy.jpg",
                file_size=1024,
                mime_type="image/jpeg",
            )

    def test_indexing_fields(self):
        """Test updating indexing status."""
        v_id = uuid.uuid4()
        now = timezone.now()

        self.media.is_indexed = True
        self.media.vector_id = v_id
        self.media.indexed_at = now
        self.media.save()

        updated_media = IndexedMedia.objects.get(id=self.media.id)
        self.assertTrue(updated_media.is_indexed)
        self.assertEqual(updated_media.vector_id, v_id)
        # Compare timestamps to avoid microsecond precision issues in some DBs
        self.assertEqual(updated_media.indexed_at.timestamp(), now.timestamp())
