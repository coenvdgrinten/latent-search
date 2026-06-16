from typing import override

from django.core.management.base import BaseCommand

from latent_search.server.indexing.services.vector_db import VectorDBService


class Command(BaseCommand):
    help = "Initializes the necessary Qdrant collections."

    @override
    def handle(self, *args, **options):
        self.stdout.write("Ensuring Qdrant collection exists...")
        service = VectorDBService()

        try:
            service.ensure_collection()
            self.stdout.write(
                self.style.SUCCESS(f"Collection '{service.collection_name}' is ready.")
            )
        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to connect or configure Qdrant: {e}")
            )
