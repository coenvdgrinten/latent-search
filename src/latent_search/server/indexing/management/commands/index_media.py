from typing import override

from django.core.management.base import BaseCommand, CommandParser
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
        service.run_discovery(path)

        self.stdout.write("Starting indexing of pending items...")
        service.index_pending_media(batch_size=batch_size)

        self.stdout.write(self.style.SUCCESS("Finished indexing run."))
