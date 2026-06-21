from typing import override

from django.core.management.base import BaseCommand, CommandParser

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.indexing import IndexingService


class Command(BaseCommand):
    help = "Discover and index media files"

    @override
    def add_arguments(self, parser: CommandParser):
        parser.add_argument("path", type=str, help="Root path to scan for media")
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of items to index in this run",
        )

    @override
    def handle(self, *args, **options):
        path = options["path"]
        batch_size = options["batch_size"]

        service = IndexingService()

        self.stdout.write(f"Starting discovery in {path}...")
        discovered = service.run_discovery(path)
        self.stdout.write(f"Discovered {discovered} new files.")

        pending_count = IndexedMedia.objects.filter(is_indexed=False).count()
        self.stdout.write(
            f"Starting indexing ({pending_count} pending, "
            f"batch size {batch_size})..."
        )
        indexed, errors = service.index_pending_media(batch_size=batch_size)

        if errors:
            self.stderr.write(f"  Errors: {errors}")
        self.stdout.write(
            self.style.SUCCESS(f"Done. Indexed {indexed}/{pending_count} items.")
        )
