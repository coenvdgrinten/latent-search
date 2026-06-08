from typing import override

from django.db import models


class IndexedMedia(models.Model):
    """
    Represents a media file found in the Nextcloud library.
    """

    file_path = models.TextField(
        unique=True, help_text="Absolute path to the media file"
    )
    filename = models.CharField(max_length=255)
    relative_path = models.TextField(help_text="Path relative to the Nextcloud root")

    # Metadata
    file_size = models.BigIntegerField(help_text="File size in bytes")
    mime_type = models.CharField(max_length=100)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    # Media-specific metadata
    taken_at = models.DateTimeField(null=True, blank=True)
    width = models.IntegerField(null=True, blank=True)
    height = models.IntegerField(null=True, blank=True)
    latitude = models.FloatField(null=True, blank=True)
    longitude = models.FloatField(null=True, blank=True)

    # Embedding status
    is_indexed = models.BooleanField(default=False)
    indexed_at = models.DateTimeField(null=True, blank=True)
    vector_id = models.UUIDField(
        null=True, blank=True, help_text="ID of the vector in Qdrant"
    )
    caption = models.TextField(
        default="",
        blank=True,
        help_text="Text caption used to generate the embedding",
    )

    class Meta:
        verbose_name = "Indexed Media"
        verbose_name_plural = "Indexed Media"
        indexes = [
            models.Index(fields=["is_indexed"]),
            models.Index(fields=["taken_at"]),
        ]

    @override
    def __str__(self):
        return f"{self.filename} ({'Indexed' if self.is_indexed else 'Pending'})"
