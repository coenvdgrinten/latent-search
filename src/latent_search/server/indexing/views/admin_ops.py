"""Admin / operations API endpoints for triggering & monitoring background jobs."""

import json
import logging
import threading
from typing import Any

from django.http import HttpRequest, JsonResponse
from django.views.decorators.csrf import csrf_exempt

from latent_search.server.indexing.models.media import IndexedMedia
from latent_search.server.indexing.services.indexing import IndexingService
from latent_search.server.indexing.services.job_manager import (
    JobKind,
    JobStatus,
    job_manager,
)
from latent_search.server.indexing.services.vlm import VLMService
from latent_search.server.indexing.views.decorators import login_required_json

logger = logging.getLogger(__name__)

# Lazily-created services – expensive to instantiate needlessly.
_indexing_svc: IndexingService | None = None
_vlm_svc: VLMService | None = None


def _get_indexing_service() -> IndexingService:
    global _indexing_svc
    if _indexing_svc is None:
        _indexing_svc = IndexingService()
    return _indexing_svc


def _get_vlm_service() -> VLMService:
    global _vlm_svc
    if _vlm_svc is None:
        _vlm_svc = VLMService()
    return _vlm_svc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stats() -> dict[str, Any]:
    """Return high-level library statistics."""
    total = IndexedMedia.objects.count()
    indexed = IndexedMedia.objects.filter(is_indexed=True).count()
    enriched = (
        IndexedMedia.objects.exclude(vlm_caption="")
        .filter(vlm_caption__isnull=False)
        .exclude(vlm_caption="")
    )
    enriched_count = enriched.count()
    return {
        "total": total,
        "indexed": indexed,
        "enriched": enriched_count,
        "pending_index": total - indexed,
        "pending_enrich": total - enriched_count,
    }


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@login_required_json
def get_stats(_request: HttpRequest) -> JsonResponse:
    """GET /api/stats – return library + job statuses."""
    return JsonResponse(
        {
            "library": _stats(),
            "jobs": job_manager.all_jobs(),
            "media_root": job_manager.media_root,
        }
    )


@csrf_exempt
@login_required_json
def start_job(request: HttpRequest) -> JsonResponse:
    """POST /api/start_job?kind=indexing&root=/path/to/media

    Kicks off a background thread for the named job kind.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    kind_str = request.GET.get("kind", "")
    try:
        kind = JobKind(kind_str)
    except ValueError:
        return JsonResponse({"error": f"Unknown job kind: {kind_str}"}, status=400)

    job = job_manager.get_job(kind)

    # Guard against double-start
    if job.status == JobStatus.RUNNING:
        return JsonResponse({"error": f"{kind.value} is already running"}, status=409)

    root_arg = request.GET.get("root", "").strip() or job_manager.media_root

    # Validate root for discovery/indexing
    if kind in (JobKind.DISCOVERY, JobKind.INDEXING) and not root_arg:
        return JsonResponse(
            {"error": "Missing 'root' query param; set media_root first"}, status=400
        )

    if kind in (JobKind.DISCOVERY, JobKind.INDEXING):
        job_manager.save_media_root(root_arg)

    # Wire up the target function
    def _run_discovery():
        svc = _get_indexing_service()
        job.status = JobStatus.RUNNING
        job.message = "Scanning filesystem…"

        def _cb(current: int, total: int, message: str) -> None:
            job_update(job, current, total, message)

        try:
            count = svc.run_discovery(root_arg, callback=_cb)
            job.total = count
            job.progress = count
            job.message = f"Discovered {count} new files"
            job.status = JobStatus.DONE
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.ERROR
            logger.exception("Discovery failed")

    def _run_indexing():
        svc = _get_indexing_service()
        job.status = JobStatus.RUNNING
        job.message = "Generating embeddings…"
        try:
            pending_qs = IndexedMedia.objects.filter(is_indexed=False)
            pending_items = list(pending_qs)
            job.total = len(pending_items)

            def _idx_cb(current: int, total: int, message: str) -> None:
                job_update(job, current, total, message)

            indexed, errors = svc.index_pending_media(
                batch_size=len(pending_items) or 10_000,
                callback=_idx_cb,
            )
            job.progress += indexed
            job.message = f"Indexed {indexed}, errors {errors}"
            job.status = JobStatus.DONE
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.ERROR
            logger.exception("Indexing failed")

    def _run_enrichment():
        svc = _get_vlm_service()
        job.status = JobStatus.RUNNING
        job.message = "Generating captions…"
        try:
            pending_qs = IndexedMedia.objects.filter(vlm_caption__exact="")
            pending_items = list(pending_qs)
            job.total = len(pending_items)
            count = 0
            error_count = 0
            for media in pending_items:
                if job.should_stop:
                    job.message = "Cancelled by user"
                    job.status = JobStatus.IDLE
                    return
                try:
                    caption = svc.describe(media.file_path)
                    media.vlm_caption = caption
                    media.save(update_fields=["vlm_caption"])
                    count += 1
                except Exception as exc:
                    logger.warning("Failed to enrich %s: %s", media.file_path, exc)
                    error_count += 1
                msg = f"Caption {count}/{len(pending_items)}"
                job_update(job, count, len(pending_items), msg)
            job.progress = count
            job.message = f"Enriched {count}, errors {error_count}"
            job.status = JobStatus.DONE
        except Exception as exc:
            job.error = str(exc)
            job.status = JobStatus.ERROR
            logger.exception("Enrichment failed")

    runners = {
        JobKind.DISCOVERY: _run_discovery,
        JobKind.INDEXING: _run_indexing,
        JobKind.ENRICHMENT: _run_enrichment,
    }

    target = runners[kind]
    job.reset()
    threading.Thread(target=target, daemon=True).start()
    return JsonResponse({"status": "started", "kind": kind.value})


@csrf_exempt
@login_required_json
def stop_job(request: HttpRequest) -> JsonResponse:
    """POST /api/stop_job?kind=indexing"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    kind_str = request.GET.get("kind", "")
    try:
        kind = JobKind(kind_str)
    except ValueError:
        return JsonResponse({"error": f"Unknown job kind: {kind_str}"}, status=400)

    job = job_manager.get_job(kind)
    if job.status != JobStatus.RUNNING:
        return JsonResponse({"error": "Nothing to stop"}, status=400)

    job.cancel()
    job.status = JobStatus.IDLE
    job.message = "Stopping …"
    return JsonResponse({"status": "stopping"})


@csrf_exempt
@login_required_json
def save_settings(request: HttpRequest) -> JsonResponse:
    """POST /api/save_settings body: {"media_root": "/path"}"""
    if request.method != "POST":
        return JsonResponse({"error": "Method not allowed"}, status=405)

    data = request.body.decode()
    try:
        payload = json.loads(data)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON"}, status=400)

    root = payload.get("media_root", "")
    job_manager.save_media_root(root)
    return JsonResponse({"status": "saved", "media_root": job_manager.media_root})


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def job_update(job, current: int, total: int, message: str) -> None:
    """Atomically bump counters on a running job."""
    job.progress = current
    job.total = total
    job.message = message
