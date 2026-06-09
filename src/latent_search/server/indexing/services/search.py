from pathlib import Path

from django.conf import settings
from httpx import ConnectError
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import (
    DatetimeRange,
    FieldCondition,
    Filter,
    MatchText,
    Prefetch,
)

from latent_search.server.indexing.services.query_parser import (
    ParsedQuery,
    parse_query,
)
from latent_search.server.indexing.services.text_embedding import (
    TextEmbeddingService,
)


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

    def _build_payload_filter(self, parsed: ParsedQuery) -> Filter | None:
        """Build a Qdrant payload filter from parsed query entities."""
        from datetime import date

        must_conditions: list[FieldCondition] = []
        should_conditions: list[FieldCondition] = []

        if parsed.year is not None:
            must_conditions.append(
                FieldCondition(
                    key="taken_at",
                    match=MatchText(text=str(parsed.year)),
                )
            )

        if parsed.month is not None:
            # ISO month format: "MM" — match within taken_at string
            month_str = f"{parsed.month:02d}"
            must_conditions.append(
                FieldCondition(key="taken_at", match=MatchText(text=month_str))
            )

        if parsed.season_months is not None:
            # Build date range conditions for the season
            # Seasons can span year boundaries (winter: Dec-Feb)
            month_list = sorted(parsed.season_months)
            if parsed.year is not None:
                # Specific year + season (e.g., "summer 2018")
                for m in month_list:
                    next_month = m + 1 if m < 12 else 1
                    next_year = parsed.year + 1 if m == 12 else parsed.year
                    should_conditions.append(
                        FieldCondition(
                            key="taken_at",
                            range=DatetimeRange(
                                gte=date(parsed.year, m, 1),
                                lt=date(next_year, next_month, 1),
                            ),
                        )
                    )
            else:
                # Generic season — match any year
                for m in month_list:
                    should_conditions.append(
                        FieldCondition(
                            key="taken_at",
                            match=MatchText(text=f"-{m:02d}-"),
                        )
                    )

        if parsed.location_keyword:
            # Only apply location filter when we also have strict date filters.
            # Without date constraints, let the embedding handle location matching
            # — filtering would exclude results where the caption uses different
            # wording (e.g., "éire" vs "ireland").
            if must_conditions:
                should_conditions.append(
                    FieldCondition(
                        key="caption",
                        match=MatchText(text=parsed.location_keyword),
                    )
                )

        if not must_conditions and not should_conditions:
            return None

        must = must_conditions if must_conditions else None
        should = should_conditions if should_conditions else None
        return Filter(must=must, should=should)  # ty: ignore[invalid-argument-type]

    def semantic_search(self, query: str, limit: int = 24) -> list[dict]:
        """
        Converts a text string to an embedding and searches both the image
        and text vectors in Qdrant, fusing results with Reciprocal Rank
        Fusion (RRF). Applies payload filters derived from extracted
        structured entities (dates, locations).

        - The image vector catches visually matching photos (scenes, textures,
          composition) even when filenames lack context.
        - The text vector catches semantic matches from enriched captions
          (filenames, locations, timestamps).
        - Payload filters narrow results by year, season, or location.
        - RRF merges both ranked lists into a single coherent ranking.
        """
        parsed = parse_query(query)
        query_embedding = self.text_embedding.encode(query)

        payload_filter = self._build_payload_filter(parsed)

        try:
            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=[
                    Prefetch(
                        query=query_embedding,
                        using="image",
                        limit=limit,
                        filter=payload_filter,
                    ),
                ],
                query=query_embedding,
                using="text",
                limit=limit,
                query_filter=payload_filter,
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
