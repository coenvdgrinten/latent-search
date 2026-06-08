import logging
import mimetypes
import re
import uuid
from pathlib import Path

from django.db import transaction

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.clip import CLIPService
from latent_search.server.indexing.services.discovery import DiscoveryService
from latent_search.server.indexing.services.exif import ExifService
from latent_search.server.indexing.services.geocoding import GeocodingService
from latent_search.server.indexing.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(self):
        self.discovery = DiscoveryService()
        self.clip = CLIPService()
        self.vector_db = VectorDBService()
        self.exif = ExifService()
        self.geocoding = GeocodingService()

    def run_discovery(self, root_path: str | Path):
        """
        Walk the filesystem and populate the database with new media records.
        """
        root = Path(root_path).absolute()
        for path in self.discovery.discover_media(root):
            abs_path = str(path.absolute())

            # Basic stats for initial record
            try:
                stat = path.stat()
                file_size = stat.st_size
            except OSError:
                file_size = 0

            # Split long line for PEP 8
            if root in path.parents or path == root:
                rel_path = str(path.relative_to(root))
            else:
                rel_path = path.name

            mime_type, _ = mimetypes.guess_type(path.name)
            meta = self.exif.read_metadata(path)

            # Using get_or_create to avoid duplicates by path
            IndexedMedia.objects.get_or_create(
                file_path=abs_path,
                defaults={
                    "filename": path.name,
                    "relative_path": rel_path,
                    "file_size": file_size,
                    "mime_type": mime_type or "",
                    "taken_at": meta.taken_at,
                    "width": meta.width,
                    "height": meta.height,
                    "latitude": meta.latitude,
                    "longitude": meta.longitude,
                    "is_indexed": False,
                },
            )

    def index_pending_media(self, batch_size: int = 100):
        """
        Process media that hasn't been indexed yet.
        """
        self.vector_db.ensure_collection()

        pending = IndexedMedia.objects.filter(is_indexed=False)[:batch_size]

        for media in pending:
            try:
                logger.info(f"Indexing {media.file_path}")
                image_embedding = self.clip.get_image_embedding(media.file_path)

                text_caption = self._build_text_caption(media)
                text_embedding = self.clip.get_text_embedding(text_caption, task=None)

                # Assign vector_id at indexing time if not set
                if not media.vector_id:
                    media.vector_id = uuid.uuid4()

                payload = {
                    "file_name": media.filename,
                    "file_path": media.file_path,
                    "taken_at": media.taken_at.isoformat() if media.taken_at else None,
                    "width": media.width,
                    "height": media.height,
                    "latitude": media.latitude,
                    "longitude": media.longitude,
                }

                self.vector_db.upsert_embedding(
                    vector_id=media.vector_id,
                    image_embedding=image_embedding,
                    text_embedding=text_embedding,
                    payload={"caption": text_caption, **payload},
                )

                with transaction.atomic():
                    media.is_indexed = True
                    media.caption = text_caption
                    media.save()

            except Exception as e:
                logger.error(f"Failed to index {media.file_path}: {e}")
                continue

    @staticmethod
    def _build_temporal_context(media: IndexedMedia) -> str | None:
        """Derive human-readable temporal descriptors from taken_at."""
        if not media.taken_at:
            return None

        month_names = [
            "january",
            "february",
            "march",
            "april",
            "may",
            "june",
            "july",
            "august",
            "september",
            "october",
            "november",
            "december",
        ]
        season = (
            "summer"
            if media.taken_at.month in (6, 7, 8)
            else "autumn"
            if media.taken_at.month in (9, 10, 11)
            else "winter"
            if media.taken_at.month in (12, 1, 2)
            else "spring"
        )

        hour = media.taken_at.hour
        time_of_day = (
            "early morning"
            if 5 <= hour < 8
            else "morning"
            if 8 <= hour < 12
            else "afternoon"
            if 12 <= hour < 17
            else "evening"
            if 17 <= hour < 21
            else "night"
        )

        month_word = month_names[media.taken_at.month - 1]
        return f"{month_word} {media.taken_at.year} {season} {time_of_day}"

    def _build_text_caption(self, media: IndexedMedia) -> str:
        """
        Build a searchable text description from filename, folder path,
        reverse-geocoded location, and temporal context.

        Parts are concatenated in order of importance (most semantically
        dense first) to minimise CLIP truncation risk.
        """
        segments: list[str] = []

        # 1. Filename + relative folder path keywords (avoid absolute path noise)
        path = Path(str(media.relative_path))
        stem = path.stem
        folder_parts = [p for p in path.parts[:-1] if len(p) > 1 and p != "/"]

        raw_words: list[str] = []
        for segment in [*folder_parts, stem]:
            raw_words.extend(re.split(r"[-_\s]+", segment))

        words = [w.lower() for w in raw_words if w and not w.isdigit() and len(w) > 1]
        # Deduplicate while preserving order
        path_keywords = " ".join(dict.fromkeys(words))
        if path_keywords:
            segments.append(path_keywords)

        # 2. Reverse-geocoded location
        if media.latitude is not None and media.longitude is not None:
            location = self.geocoding.reverse_geocode(
                float(media.latitude),  # ty: ignore[invalid-argument-type]
                float(media.longitude),  # ty: ignore[invalid-argument-type]
            )
            if location:
                segments.append(location.lower())

        # 3. Temporal context
        temporal = self._build_temporal_context(media)
        if temporal:
            segments.append(temporal)

        return " ".join(segments) if segments else "unknown subject"
