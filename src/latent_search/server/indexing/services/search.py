from pathlib import Path

from django.conf import settings
from httpx import ConnectError
from latent_search.server.indexing.services.text_embedding import (
    TextEmbeddingService,
)
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Prefetch


class QdrantUnavailableError(Exception):
    pass


class SearchService:
    def __init__(self) -> None:
        self.text_embedding = TextEmbeddingService()
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def _construct_image_url(self, file_path: str) -> str | None:
        """
        Construct a URL for serving the image file.

        Maps the absolute file path to a media URL based on MEDIA_ROOT.
        For example: /path/to/media/photos/img.jpg -> /media/photos/img.jpg
        """
        if not file_path:
            return None

        media_root = Path(settings.MEDIA_ROOT).resolve()
        try:
            relative = Path(file_path).resolve().relative_to(media_root)
            return f"{settings.MEDIA_URL}{relative}"
        except ValueError:
            # File is not under MEDIA_ROOT
            return None

    def semantic_search(self, query: str, limit: int = 24) -> list[dict]:
        """
        Converts a text string to an embedding and searches both the image
        and text vectors in Qdrant, fusing results with Reciprocal Rank
        Fusion (RRF).

        - The image vector catches visually matching photos (scenes, textures,
          composition) even when filenames lack context.
        - The text vector catches semantic matches from enriched captions
          (filenames, locations, timestamps).
        - RRF merges both ranked lists into a single coherent ranking.
        """
        query_embedding = self.text_embedding.encode(query)

        try:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using="image",
                        limit=limit,
                    ),
                ],
                query=query_embedding,
                using="text",
                limit=limit,
            ).points
        except (ResponseHandlingException, ConnectError) as exc:
            raise QdrantUnavailableError(
                "Could not connect to Qdrant. Is the Qdrant service running?"
            ) from exc

        hits: list[dict] = []
        for hit in search_results:
            payload = hit.payload or {}
            taken_at = payload.get("taken_at")
            lat = payload.get("latitude")
            lon = payload.get("longitude")

            # Parse date nicely
            date_display = None
            if taken_at:
                try:
                    from datetime import datetime

                    dt = datetime.fromisoformat(taken_at.replace("Z", "+00:00"))
                    date_display = dt.strftime("%b %Y")
                except (ValueError, AttributeError):
                    pass

            # Location from caption (last comma-separated segment)
            caption = payload.get("caption", "")
            location = None
            if lat is not None and lon is not None and caption:
                parts = [p.strip() for p in caption.split(",")]
                if len(parts) >= 2:
                    location = ", ".join(parts[-3:])  # City, Region, Country

            hits.append(
                {
                    "id": hit.id,
                    "score": hit.score,
                    "file_path": payload.get("file_path", ""),
                    "file_name": payload.get("file_name", ""),
                    "image_url": self._construct_image_url(
                        payload.get("file_path", "")
                    ),
                    "date_taken": date_display,
                    "location": location,
                }
            )

        return hits
