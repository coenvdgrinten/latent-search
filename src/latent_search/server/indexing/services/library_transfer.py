"""Export / import service for cross-machine library migration."""

import json
import logging
from collections.abc import Iterator
from datetime import datetime

from django.http import FileResponse, HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.views.decorators import login_required_json

logger = logging.getLogger(__name__)


# Fields exported per record (order matters for consistency).
_EXPORT_FIELDS = [
    "file_path",
    "filename",
    "relative_path",
    "file_size",
    "mime_type",
    "taken_at",
    "width",
    "height",
    "latitude",
    "longitude",
    "is_indexed",
    "indexed_at",
    "vector_id",
    "caption",
    "vlm_caption",
]

# Schema version — bump if format changes.
_SCHEMA_VERSION = 1


def _serialize_record(obj: IndexedMedia) -> dict:
    """Convert an IndexedMedia instance to a flat dict."""
    record: dict = {"version": _SCHEMA_VERSION}
    for field in _EXPORT_FIELDS:
        val = getattr(obj, field, None)
        # Normalize datetimes to ISO strings
        if isinstance(val, datetime):
            val = val.isoformat()
        # Normalize UUIDs to strings
        if hasattr(val, "hex"):
            val = str(val)
        record[field] = val
    return record


def _deserialize_line(line: str) -> dict | None:
    """Parse a single NDJSON line; skip blanks/comments."""
    stripped = line.strip()
    if not stripped or stripped.startswith("#"):
        return None
    try:
        return json.loads(stripped)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed line: %s", stripped[:80])
        return None


@login_required_json
def export_library(request: HttpRequest) -> FileResponse:
    """GET /api/export_library?[indexed_only=true]&[uncaptioned_only=false]

    Streams an NDJSON file with one IndexedMedia record per line.
    """
    qs = request.GET

    base_qs = IndexedMedia.objects.all()

    # Filter options
    if qs.get("indexed_only") == "true":
        base_qs = base_qs.filter(is_indexed=True)
    if qs.get("uncaptioned_only") == "true":
        base_qs = base_qs.filter(vlm_caption__exact="")

    # Limit to reasonable chunk sizes
    queryset = base_qs.order_by("-created_at")

    def generate_lines() -> Iterator[str]:
        yield f"# LatentSearch export — schema v{_SCHEMA_VERSION}\n"
        for obj in queryset.iterator(chunk_size=500):
            yield json.dumps(_serialize_record(obj), default=str) + "\n"

    response = FileResponse(
        generate_lines(),
        content_type="application/x-ndjson",
        as_attachment=True,
        filename=f"latent_export_{datetime.now():%Y%m%d_%H%M%S}.ndjson",
    )
    return response


@csrf_exempt
@login_required_json
def import_library(request: HttpRequest) -> JsonResponse:
    """POST /api/import_library

    Upload an NDJSON file produced by /api/export_library.
    Upserts records keyed by file_path.

    Merge strategy: incoming non-null/non-empty values overwrite existing rows.
    This lets you round-trip partial updates (e.g., only captions changed).
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    file = request.FILES.get("file")
    if not file:
        return JsonResponse({"error": "No file attached"}, status=400)

    created = 0
    updated = 0
    errors = 0
    total = 0

    seen_paths: set[str] = set()

    # Handle both InMemoryUploadedFile and TemporaryUploadedFile
    raw: bytes | str = file.read()
    decoded = raw.decode("utf-8") if isinstance(raw, bytes) else raw

    for line in decoded.splitlines():
        record = _deserialize_line(line)
        if record is None:
            continue
        total += 1

        fp = record.get("file_path")
        if not fp:
            errors += 1
            continue

        try:
            _, is_created = IndexedMedia.objects.update_or_create(
                file_path=fp,
                defaults=_clean_defaults(record),
            )
            if is_created:
                created += 1
            else:
                updated += 1
            seen_paths.add(fp)
        except Exception as exc:
            logger.warning("Failed to import %s: %s", fp, exc)
            errors += 1

    return JsonResponse(
        {
            "status": "ok",
            "total_lines": total,
            "created": created,
            "updated": updated,
            "errors": errors,
        }
    )


def _clean_defaults(record: dict) -> dict:
    """Build a defaults dict for update_or_create, skipping sentinel empties."""
    defaults: dict = {}

    # Nullable numeric/date fields — treat None/null as intentional
    nullable = {
        "taken_at",
        "width",
        "height",
        "latitude",
        "longitude",
        "indexed_at",
        "vector_id",
    }

    for field in _EXPORT_FIELDS:
        if field == "file_path":
            continue
        val = record.get(field)

        if field in nullable:
            # Preserve explicit NULLs (they might be meaningful resets)
            if val is not None:
                # Parse ISO dates back to datetime
                if field in ("taken_at", "indexed_at") and isinstance(val, str):
                    try:
                        val = datetime.fromisoformat(val)
                    except ValueError:
                        pass
                # Parse UUID strings
                if field == "vector_id" and isinstance(val, str):
                    import uuid

                    try:
                        val = uuid.UUID(val)
                    except ValueError:
                        pass
                defaults[field] = val
        else:
            # Non-nullable: skip empty strings/falsey booleans unless explicitly sent
            if field == "is_indexed":
                defaults["is_indexed"] = bool(val)
            elif field in ("caption", "vlm_description"):
                # Allow clearing captions intentionally
                defaults[field] = val if val is not None else ""
            elif field in ("caption", "vlm_caption"):
                if val is not None:
                    defaults[field] = val
            elif val:
                defaults[field] = val

    return defaults
