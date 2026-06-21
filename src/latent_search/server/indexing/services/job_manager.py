"""
Background job runner for indexing and enrichment operations.

Provides a singleton that manages threaded jobs, tracks progress,
and exposes status for HTMX polling from the UI.
"""

import logging
import threading
from enum import Enum
from typing import Any

from django.conf import settings

logger = logging.getLogger(__name__)


class JobKind(Enum):
    DISCOVERY = "discovery"
    INDEXING = "indexing"
    ENRICHMENT = "enrichment"


class JobStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DONE = "done"
    ERROR = "error"


class JobInfo:
    """Mutable container holding live state of a single job kind."""

    KIND: JobKind | None = None

    def __init__(self) -> None:
        self.status = JobStatus.IDLE
        self.progress = 0
        self.total = 0
        self.message = ""
        self.error: str | None = None
        self._stop_event = threading.Event()

    @property
    def should_stop(self) -> bool:
        return self._stop_event.is_set()

    def cancel(self) -> None:
        self._stop_event.set()

    def reset(self) -> None:
        self.status = JobStatus.IDLE
        self.progress = 0
        self.total = 0
        self.message = ""
        self.error = None
        self._stop_event.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.KIND.value if self.KIND else "",
            "status": self.status.value,
            "progress": self.progress,
            "total": self.total,
            "message": self.message,
            "error": self.error,
        }


class _DiscoveryJob(JobInfo):
    KIND = JobKind.DISCOVERY


class _IndexingJob(JobInfo):
    KIND = JobKind.INDEXING


class _EnrichmentJob(JobInfo):
    KIND = JobKind.ENRICHMENT


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

class JobManager:
    """Central registry for all background indexing/enrichment jobs."""

    def __init__(self) -> None:
        self.discovery = _DiscoveryJob()
        self.indexing = _IndexingJob()
        self.enrichment = _EnrichmentJob()
        self.media_root: str = getattr(settings, "LATENT_MEDIA_ROOT", "")

    def get_job(self, kind: JobKind) -> JobInfo:
        mapping = {
            JobKind.DISCOVERY: self.discovery,
            JobKind.INDEXING: self.indexing,
            JobKind.ENRICHMENT: self.enrichment,
        }
        return mapping[kind]

    def all_jobs(self) -> list[dict[str, Any]]:
        return [
            self.discovery.to_dict(),
            self.indexing.to_dict(),
            self.enrichment.to_dict(),
        ]

    def save_media_root(self, path: str) -> None:
        """Persist the configured media root between requests."""
        self.media_root = path.strip()


job_manager = JobManager()
