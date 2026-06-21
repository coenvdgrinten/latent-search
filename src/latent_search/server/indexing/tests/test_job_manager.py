"""Tests for the background job manager."""

from typing import override
from unittest import TestCase

from latent_search.server.indexing.services.job_manager import (
    JobKind,
    JobStatus,
    job_manager,
)


class JobManagerTest(TestCase):
    @override
    def setUp(self):
        # Reset all jobs before each test
        targets = [
            job_manager.discovery,
            job_manager.indexing,
            job_manager.enrichment,
        ]
        for job in targets:
            job.reset()
        job_manager.media_root = ""

    def test_default_states_are_idle(self):
        assert job_manager.discovery.status == JobStatus.IDLE
        assert job_manager.indexing.status == JobStatus.IDLE
        assert job_manager.enrichment.status == JobStatus.IDLE

    def test_all_jobs_returns_three_entries(self):
        result = job_manager.all_jobs()
        assert len(result) == 3

    def test_all_jobs_contains_correct_kinds(self):
        kinds = [j["kind"] for j in job_manager.all_jobs()]
        assert "discovery" in kinds
        assert "indexing" in kinds
        assert "enrichment" in kinds

    def test_to_dict_serializes_fields(self):
        job = job_manager.indexing
        job.status = JobStatus.RUNNING
        job.progress = 42
        job.total = 100
        job.message = "Halfway there"

        d = job.to_dict()
        assert d["kind"] == "indexing"
        assert d["status"] == "running"
        assert d["progress"] == 42
        assert d["total"] == 100
        assert d["message"] == "Halfway there"
        assert d["error"] is None

    def test_cancel_sets_should_stop_flag(self):
        job = job_manager.discovery
        assert not job.should_stop
        job.cancel()
        assert job.should_stop

    def test_reset_clears_everything(self):
        job = job_manager.enrichment
        job.status = JobStatus.DONE
        job.progress = 50
        job.total = 100
        job.message = "Finished"
        job.error = "oops"
        job.cancel()

        job.reset()

        assert job.status == JobStatus.IDLE
        assert job.progress == 0
        assert job.total == 0
        assert job.message == ""
        assert job.error is None
        assert not job.should_stop

    def test_save_media_root_persists(self):
        job_manager.save_media_root("/path/to/photos")
        assert job_manager.media_root == "/path/to/photos"

    def test_get_job_by_kind(self):
        retrieved = job_manager.get_job(JobKind.INDEXING)
        assert retrieved is job_manager.indexing
