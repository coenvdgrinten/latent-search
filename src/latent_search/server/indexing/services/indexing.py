import logging
import mimetypes
import uuid
from pathlib import Path

from django.db import transaction

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.clip import CLIPService
from latent_search.server.indexing.services.discovery import DiscoveryService
from latent_search.server.indexing.services.exif import ExifService
from latent_search.server.indexing.services.vector_db import VectorDBService

logger = logging.getLogger(__name__)


class IndexingService:
    def __init__(self):
        self.discovery = DiscoveryService()
        self.clip = CLIPService()
        self.vector_db = VectorDBService()
        self.exif = ExifService()

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
                embedding = self.clip.get_image_embedding(media.file_path)

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
                    vector_id=media.vector_id, embedding=embedding, payload=payload
                )

                with transaction.atomic():
                    media.is_indexed = True
                    media.save()

            except Exception as e:
                logger.error(f"Failed to index {media.file_path}: {e}")
                continue
