"""Django management command to run the folder‑watcher service.

Usage:
    ./manage watch_media

The command blocks and monitors ``settings.MEDIA_ROOT`` for new image files.
It is deliberately lightweight – all heavy lifting (embedding, captioning,
Qdrant upserts) is delegated to the existing ``IndexingService``.
"""

from typing import override

from django.core.management.base import BaseCommand

from latent_search.server.indexing.services.folder_watcher import start_watcher


class Command(BaseCommand):
    help = "Continuously watch MEDIA_ROOT for new images and auto‑index them."

    @override
    def handle(self, *args, **options):  # pragma: no cover – integration command
        start_watcher()
