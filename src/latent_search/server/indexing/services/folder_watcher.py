"""Folder‑watcher service for automatic indexing.

This module uses the *watchdog* library to monitor ``settings.MEDIA_ROOT`` for
new image files. When a supported file appears it triggers the existing
``IndexingService`` to discover the file and index any pending media.

The watcher is started via the ``watch_media`` management command (see the
corresponding file added below). It runs as a long‑living process and can be
deployed as a side‑car container, systemd service, etc.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import override

from django.conf import settings
from watchdog.events import (
    DirCreatedEvent,
    DirMovedEvent,
    FileCreatedEvent,
    FileMovedEvent,
    FileSystemEventHandler,
)
from watchdog.observers import Observer

from latent_search.server.indexing.services.indexing import IndexingService

logger = logging.getLogger(__name__)

# Configuration – extensions considered as media images. Keep this in sync
# with ``DiscoveryService`` if that list ever changes.
# Use the built‑in ``set`` for the type hint (PEP 585) – ``typing.Set`` is
# deprecated in modern Python.
IMAGE_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".tif", ".tiff"}


class MediaEventHandler(FileSystemEventHandler):
    """Handle only the events we care about – creation or move of an image file."""

    def __init__(self, indexing_srv: IndexingService) -> None:
        self.indexing_srv = indexing_srv
        super().__init__()

    # Helper utilities
    @staticmethod
    def _is_image(path: Path) -> bool:
        return path.suffix.lower() in IMAGE_EXTS

    def _process_path(self, path: Path) -> None:
        """Run discovery + indexing for a single image path.

        ``IndexingService.run_discovery`` walks the *parent* directory, which
        ensures the new file is added to the DB (or ignored if it already
        exists). Afterwards we call ``index_pending_media`` with a small batch
        size so the newly‑added record is processed promptly.
        """
        if not self._is_image(path):
            return
        logger.info("New media detected: %s – triggering indexing", path)
        # Discover the file (adds a DB row if missing)
        self.indexing_srv.run_discovery(path.parent)
        # Index any pending items – the new file will be among them
        self.indexing_srv.index_pending_media(batch_size=10)

    # Event callbacks – we only care about files, not directories
    @override
    def on_created(
        self, event: FileCreatedEvent | DirCreatedEvent
    ) -> None:  # pragma: no cover
        # ``event`` may be a directory creation – ignore those.
        if not getattr(event, "is_directory", False):
            self._process_path(Path(str(event.src_path)))

    @override
    def on_moved(
        self, event: FileMovedEvent | DirMovedEvent
    ) -> None:  # pragma: no cover
        if not getattr(event, "is_directory", False):
            self._process_path(Path(str(event.dest_path)))


def start_watcher() -> None:
    """Entry point used by the ``watch_media`` management command.

    The function respects ``settings.MEDIA_WATCH_ENABLED`` – if the flag is
    ``False`` the watcher simply logs a warning and returns.
    """
    if not getattr(settings, "MEDIA_WATCH_ENABLED", False):
        logger.warning("Media‑watcher is disabled (MEDIA_WATCH_ENABLED=False)")
        return

    media_root = Path(settings.MEDIA_ROOT).resolve()
    if not media_root.is_dir():
        logger.error("MEDIA_ROOT does not exist or is not a directory: %s", media_root)
        return

    indexing_srv = IndexingService()
    handler = MediaEventHandler(indexing_srv)

    observer = Observer()
    observer.schedule(handler, str(media_root), recursive=True)
    observer.start()
    logger.info("Started folder‑watcher on %s", media_root)

    try:
        observer.join()
    except KeyboardInterrupt:  # pragma: no cover – graceful shutdown
        observer.stop()
    observer.join()
