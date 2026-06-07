import os
from typing import override

from django.core.management.base import BaseCommand
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class Command(BaseCommand):
    help = "Initializes the necessary Qdrant collections."

    @override
    def handle(self, *args, **options):
        host = os.environ.get("QDRANT_HOST", "localhost")
        port = int(os.environ.get("QDRANT_PORT", 6333))

        self.stdout.write(f"Connecting to Qdrant at {host}:{port}...")
        client = QdrantClient(host=host, port=port)

        collection_name = "nextcloud_media"

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
