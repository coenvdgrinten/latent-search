"""
Management command to offload indexing work to a remote GPU machine.

Pulls pending records from a LatentSearch instance over HTTP, downloads
each image, runs VLM captioning + CLIP/BGE embedding locally, upserts
vectors directly to Qdrant, and writes captions + is_indexed flags back
to the LatentSearch instance via the import_library endpoint.

Designed for the cross-machine workflow described in OFFLOAD.md:
the Unraid box stays low-power, the GPU box does the heavy ML work.
"""

import io
import json
import logging
import tempfile
import time
import uuid
from pathlib import Path
from typing import override

import requests
from django.core.management.base import BaseCommand, CommandError, CommandParser
from tqdm import tqdm

from latent_search.server.indexing.services.clip import CLIPService
from latent_search.server.indexing.services.geocoding import GeocodingService
from latent_search.server.indexing.services.indexing import IndexingService
from latent_search.server.indexing.services.sparse_encoding import (
    SparseEncodingService,
)
from latent_search.server.indexing.services.text_embedding import (
    TextEmbeddingService,
)
from latent_search.server.indexing.services.vector_db import VectorDBService
from latent_search.server.indexing.services.vlm import VLMService

logger = logging.getLogger(__name__)


class Command(BaseCommand):
    help = (
        "Offload indexing to this (GPU-capable) machine by pulling pending "
        "records from a remote LatentSearch instance and pushing vectors "
        "directly to its Qdrant."
    )

    @override
    def add_arguments(self, parser: CommandParser):
        parser.add_argument(
            "--api-url",
            type=str,
            required=True,
            help="Base URL of the remote LatentSearch instance (e.g. http://unraid:8000)",
        )
        parser.add_argument(
            "--api-user",
            type=str,
            required=True,
            help="Username on the remote LatentSearch instance",
        )
        parser.add_argument(
            "--api-password",
            type=str,
            required=True,
            help="Password on the remote LatentSearch instance",
        )
        parser.add_argument(
            "--qdrant-url",
            type=str,
            default=None,
            help=(
                "Qdrant URL. Defaults to the remote LatentSearch's host on "
                "port 6333 (e.g. http://unraid:6333)."
            ),
        )
        parser.add_argument(
            "--qdrant-api-key",
            type=str,
            default=None,
            help="Qdrant API key (if the remote Qdrant requires one).",
        )
        parser.add_argument(
            "--qdrant-collection",
            type=str,
            default="media_embeddings",
            help="Qdrant collection name (must match the remote instance).",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=50,
            help="Number of records to process before posting a write-back batch.",
        )
        parser.add_argument(
            "--limit",
            type=int,
            default=None,
            help="Maximum number of records to process in this run.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Do everything except the Qdrant upsert and the import POST.",
        )
        parser.add_argument(
            "--reprocess",
            action="store_true",
            help="Process already-indexed records too (rebuild vectors).",
        )
        parser.add_argument(
            "--skip-vlm",
            action="store_true",
            help="Skip VLM captioning; embed from existing captions only.",
        )

    @override
    def handle(self, *args, **options):
        api_url = options["api_url"].rstrip("/")
        dry_run = options["dry_run"]
        batch_size = options["batch_size"]
        limit = options["limit"]
        reprocess = options["reprocess"]
        skip_vlm = options["skip_vlm"]

        # ── Resolve Qdrant URL ──
        qdrant_url = options["qdrant_url"]
        if qdrant_url is None:
            # Derive from api_url host
            from urllib.parse import urlparse

            parsed = urlparse(api_url)
            qdrant_url = f"{parsed.scheme}://{parsed.hostname}:6333"

        # ── Configure Qdrant via env so VectorDBService picks it up ──
        import os

        os.environ["QDRANT_URL"] = qdrant_url
        if options["qdrant_api_key"]:
            os.environ["QDRANT_API_KEY"] = options["qdrant_api_key"]
        os.environ["QDRANT_COLLECTION"] = options["qdrant_collection"]

        # ── Auth session with the remote LatentSearch ──
        session = requests.Session()
        self.stdout.write(f"Logging in to {api_url} ...")
        login_resp = session.post(
            f"{api_url}/login/",
            data={"username": options["api_user"], "password": options["api_password"]},
            allow_redirects=False,
        )
        # Django login redirects on success (302) or returns 200 with form errors.
        if login_resp.status_code not in (200, 302):
            raise CommandError(
                f"Login failed (status {login_resp.status_code}): "
                f"{login_resp.text[:200]}"
            )
        # Verify we're actually authenticated by hitting a protected endpoint.
        probe = session.get(f"{api_url}/api/stats")
        if probe.status_code == 401:
            raise CommandError(
                "Login did not authenticate — check --api-user / --api-password."
            )
        self.stdout.write(self.style.SUCCESS("Authenticated."))

        # ── Fetch pending records ──
        export_params = {}
        if not reprocess:
            export_params["pending_only"] = "true"
        self.stdout.write(f"Fetching record list from {api_url}/api/export_library ...")
        export_resp = session.get(
            f"{api_url}/api/export_library", params=export_params, stream=True
        )
        export_resp.raise_for_status()

        records: list[dict] = []
        for raw in export_resp.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8")
            if line.startswith("#"):
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                logger.warning("Skipping malformed export line: %s", line[:80])

        if limit:
            records = records[:limit]

        total = len(records)
        if total == 0:
            self.stdout.write(self.style.WARNING("No pending records to process."))
            return

        self.stdout.write(f"Found {total} record(s) to process.")

        # ── Initialise services (lazy ML models) ──
        self.stdout.write("Initialising ML services (models load on first use)...")
        clip = CLIPService()
        text_embedding = TextEmbeddingService()
        sparse_encoding = SparseEncodingService()
        vlm = VLMService() if not skip_vlm else None
        geocoding = GeocodingService()
        vector_db = VectorDBService()

        if not dry_run:
            vector_db.ensure_collection()

        # ── Process ──
        processed = 0
        errors = 0
        writeback: list[dict] = []
        start_time = time.time()

        for record in tqdm(records, desc="Offloading", unit="img"):
            file_path = record.get("file_path")
            media_id = record.get("id")
            if not file_path or media_id is None:
                errors += 1
                continue

            try:
                # 1. Fetch image bytes
                img_resp = session.get(
                    f"{api_url}/api/media/{media_id}/image", stream=True
                )
                if img_resp.status_code != 200:
                    raise RuntimeError(
                        f"image fetch failed: HTTP {img_resp.status_code}"
                    )
                suffix = Path(file_path).suffix or ".img"
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    for chunk in img_resp.iter_content(chunk_size=64 * 1024):
                        tmp.write(chunk)
                    tmp_path = tmp.name

                try:
                    # 2. VLM caption (if missing and not skipped)
                    vlm_caption = record.get("vlm_caption") or ""
                    if vlm and not vlm_caption:
                        vlm_caption = vlm.describe(tmp_path)

                    # 3. Build a synthetic IndexedMedia-like object for caption builder.
                    #    We only need the fields build_text_caption reads.
                    caption_media = _CaptionMedia(record, vlm_caption)
                    caption = IndexingService.build_text_caption(
                        caption_media,
                        vlm_caption=vlm_caption or None,
                        geocoding=geocoding,
                    )

                    # 4. Embeddings
                    image_embedding = clip.get_image_embedding(tmp_path)
                    text_vec = text_embedding.encode(caption)
                    sparse_embedding = sparse_encoding.encode_document(caption)

                    # 5. Vector ID (preserve existing or mint new)
                    vector_id_str = record.get("vector_id")
                    if vector_id_str:
                        try:
                            vector_id = uuid.UUID(str(vector_id_str))
                        except (ValueError, AttributeError):
                            vector_id = uuid.uuid4()
                    else:
                        vector_id = uuid.uuid4()

                    # 6. Upsert to Qdrant
                    if not dry_run:
                        payload = {
                            "file_name": record.get("filename", ""),
                            "file_path": file_path,
                            "taken_at": record.get("taken_at"),
                            "width": record.get("width"),
                            "height": record.get("height"),
                            "latitude": record.get("latitude"),
                            "longitude": record.get("longitude"),
                            "vlm_caption": vlm_caption,
                            "caption": caption,
                        }
                        vector_db.upsert_embedding(
                            vector_id=vector_id,
                            image_embedding=image_embedding,
                            text_embedding=text_vec,
                            payload={"caption": caption, **payload},
                            sparse_embedding=sparse_embedding,
                        )

                    # 7. Queue write-back
                    writeback.append(
                        {
                            "file_path": file_path,
                            "vlm_caption": vlm_caption,
                            "caption": caption,
                            "vector_id": str(vector_id),
                            "is_indexed": True,
                        }
                    )
                    processed += 1

                finally:
                    try:
                        Path(tmp_path).unlink(missing_ok=True)
                    except OSError:
                        pass

            except Exception as exc:
                logger.error("Failed to process %s: %s", file_path, exc, exc_info=True)
                self.stderr.write(f"  error: {file_path}: {exc}")
                errors += 1

            # 8. Flush write-back batch
            if len(writeback) >= batch_size:
                self._post_writeback(session, api_url, writeback, dry_run)
                writeback.clear()

        # Final flush
        if writeback:
            self._post_writeback(session, api_url, writeback, dry_run)

        elapsed = time.time() - start_time
        self.stdout.write("")
        self.stdout.write(
            self.style.SUCCESS(
                f"Done in {elapsed / 60:.1f}m. "
                f"Processed {processed}/{total}, errors {errors}."
            )
        )

    def _post_writeback(
        self,
        session: requests.Session,
        api_url: str,
        records: list[dict],
        dry_run: bool,
    ) -> None:
        """POST an NDJSON batch back to /api/import_library."""
        if dry_run:
            self.stdout.write(
                f"  [dry-run] would write back {len(records)} record(s)."
            )
            return

        buf = io.StringIO()
        buf.write(f"# LatentSearch offload writeback — {len(records)} records\n")
        for rec in records:
            buf.write(json.dumps(rec, default=str) + "\n")

        resp = session.post(
            f"{api_url}/api/import_library",
            files={
                "file": (
                    "writeback.ndjson",
                    buf.getvalue().encode("utf-8"),
                    "application/x-ndjson",
                )
            },
        )
        if resp.status_code != 200:
            self.stderr.write(
                f"  writeback failed (HTTP {resp.status_code}): {resp.text[:200]}"
            )
        else:
            self.stdout.write(
                f"  wrote back {len(records)} record(s): {resp.json()}"
            )


class _CaptionMedia:
    """
    Lightweight stand-in for IndexedMedia used by build_text_caption.

    The offload worker doesn't have a local DB, so it can't hydrate real
    IndexedMedia instances. This duck-typed object exposes just the fields
    the caption builder reads (relative_path, latitude, longitude, taken_at),
    populated from the exported NDJSON record.
    """

    def __init__(self, record: dict, vlm_caption: str):
        self.relative_path = record.get("relative_path") or record.get("filename", "")
        self.latitude = record.get("latitude")
        self.longitude = record.get("longitude")
        self.taken_at = _parse_dt(record.get("taken_at"))
        self.vlm_caption = vlm_caption


def _parse_dt(value):
    """Parse an ISO datetime string back to a datetime, or None."""
    if not value:
        return None
    from datetime import datetime

    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None
