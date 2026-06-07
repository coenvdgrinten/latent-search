from typing import override

from django.conf import settings
from django.core.management.base import BaseCommand
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class Command(BaseCommand):
    help = "Initializes the necessary Qdrant collections."

    @override
    def handle(self, *args, **options):
        self.stdout.write(f"Connecting to Qdrant at {settings.QDRANT_URL}...")
        client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )

        collection_name = settings.QDRANT_COLLECTION

        try:
            collections = client.get_collections().collections
            exists = any(c.name == collection_name for c in collections)

            if exists:
                self.stdout.write(
                    self.style.WARNING(
                        f"Collection '{collection_name}' already exists."
                    )
                )
                return

            # Initialize collection for OpenAI CLIP (512 dimensions)
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=512, distance=Distance.COSINE),
            )
            self.stdout.write(
                self.style.SUCCESS(
                    f"Successfully created collection '{collection_name}'."
                )
            )

        except Exception as e:
            self.stdout.write(
                self.style.ERROR(f"Failed to connect or configure Qdrant: {e}")
            )
