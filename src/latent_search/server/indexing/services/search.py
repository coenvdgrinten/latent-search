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
    Rrf,
    RrfQuery,
    SparseVector,
)

from latent_search.server.indexing.services.query_parser import (
    ParsedQuery,
    parse_query,
)
from latent_search.server.indexing.services.sparse_encoding import (
    SparseEncodingService,
)
from latent_search.server.indexing.services.text_embedding import (
    TextEmbeddingService,
)
from latent_search.server.indexing.services.vector_db import SPARSE_VECTOR_NAME


class QdrantUnavailableError(Exception):
    pass


class SearchService:
    def __init__(self) -> None:
        self.text_embedding = TextEmbeddingService()
        self.sparse_encoding = SparseEncodingService()
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY,
        )
        self.collection_name = settings.QDRANT_COLLECTION

    def _construct_image_url(self, file_path: str) -> str | None:
        """
        Construct a URL for serving the image file.

        Encodes the absolute filesystem path as base64 so special characters
        (slashes, apostrophes) don't break the URL. Served via the image-proxy
        view which streams from any path on disk.
        """
        import base64

        if not file_path:
            return None

        encoded = base64.urlsafe_b64encode(file_path.encode()).decode()
        return f"/image/{encoded}/"

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

    def _hit_to_dict(self, hit) -> dict:
        """
        Convert a Qdrant search hit to a template-friendly dictionary.

        Extracts payload fields and derives display-friendly values
        (formatted date, location string, dimensions) from raw data.
        """
        from datetime import datetime

        payload = hit.payload or {}
        taken_at = payload.get("taken_at")
        lat = payload.get("latitude")
        lon = payload.get("longitude")

        # Parse date nicely
        date_display = None
        if taken_at:
            try:
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

        return {
            "id": hit.id,
            "score": hit.score,
            "file_path": payload.get("file_path", ""),
            "file_name": payload.get("file_name", ""),
            "image_url": self._construct_image_url(payload.get("file_path", "")),
            "date_taken": date_display,
            "location": location,
            "vlm_caption": payload.get("vlm_caption", ""),
            "dimensions": (
                f"{payload['width']}\u00d7{payload['height']}"
                if payload.get("width") and payload.get("height")
                else None
            ),
        }

    def _has_sparse_support(self) -> bool:
        """Check whether the current collection supports sparse vectors."""
        try:
            info = self.qdrant_client.get_collection(
                collection_name=self.collection_name
            )
            return SPARSE_VECTOR_NAME in (info.config.params.sparse_vectors or {})
        except (ResponseHandlingException, ConnectError):
            return False

    def semantic_search(self, query: str, limit: int = 24) -> list[dict]:
        """
        Triple-vector RRF search: image vector for visual matches, text
        vector for semantic caption matches, sparse vector for lexical
        keyword matches — all fused by Qdrant prefetch + RRF.

        Applies payload filters derived from structured entities
        (dates, seasons, locations) extracted by the query parser.

        Gracefully degrades to dual-vector search if the collection
        doesn't support sparse vectors yet.
        """
        parsed = parse_query(query)
        query_embedding = self.text_embedding.encode(query)

        payload_filter = self._build_payload_filter(parsed)

        try:
            # Check if collection supports sparse vectors
            has_sparse = self._has_sparse_support()

            prefetch_list: list[Prefetch] = [
                Prefetch(
                    query=query_embedding,
                    using="image",
                    limit=limit,
                    filter=payload_filter,
                ),
                Prefetch(
                    query=query_embedding,
                    using="text",
                    limit=limit,
                    filter=payload_filter,
                ),
            ]
            if has_sparse:
                sparse_result = self.sparse_encoding.encode_document(query)
                sparse_vector = SparseVector(
                    indices=sparse_result["indices"],  # ty: ignore
                    values=sparse_result["values"],  # ty: ignore
                )
                prefetch_list.append(
                    Prefetch(
                        query=sparse_vector,
                        using=SPARSE_VECTOR_NAME,
                        limit=limit,
                        filter=payload_filter,
                    )
                )

            search_results = self.qdrant_client.query_points(
                collection_name=self.collection_name,
                prefetch=prefetch_list,
                query=RrfQuery(rrf=Rrf(k=60)),
                limit=limit,
            ).points
        except (ResponseHandlingException, ConnectError) as exc:
            raise QdrantUnavailableError(
                "Could not connect to Qdrant. Is the Qdrant service running?"
            ) from exc

        return [self._hit_to_dict(hit) for hit in search_results]
