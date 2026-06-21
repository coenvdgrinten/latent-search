"""Tests for the admin operations API endpoints."""

import json

from django.test import TestCase


class AdminOpsApiTest(TestCase):
    """Test GET /api/stats and POST /api/save_settings endpoints."""

    def test_get_stats_returns_library_and_jobs(self):
        resp = self.client.get("/api/stats")
        assert resp.status_code == 200
        data = json.loads(resp.content)
        assert "library" in data
        assert "jobs" in data
        assert "media_root" in data
        assert isinstance(data["library"]["total"], int)
        assert isinstance(data["jobs"], list)
        assert len(data["jobs"]) == 3

    def test_save_settings_updates_media_root(self):
        payload = {"media_root": "/mnt/photos"}
        resp = self.client.post(
            "/api/save_settings",
            data=json.dumps(payload),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = json.loads(resp.content)
        assert data["status"] == "saved"
        assert data["media_root"] == "/mnt/photos"

    def test_save_settings_requires_post(self):
        resp = self.client.get("/api/save_settings")
        assert resp.status_code == 405

    def test_save_settings_invalid_json_fails(self):
        resp = self.client.post(
            "/api/save_settings",
            data="not-json",
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_start_job_requires_post(self):
        resp = self.client.get("/api/start_job?kind=indexing")
        assert resp.status_code == 405

    def test_start_job_unknown_kind_fails(self):
        resp = self.client.post("/api/start_job?kind=unknown")
        assert resp.status_code == 400

    def test_stop_job_requires_post(self):
        resp = self.client.get("/api/stop_job?kind=indexing")
        assert resp.status_code == 405

    def test_stop_job_unknown_kind_fails(self):
        resp = self.client.post("/api/stop_job?kind=unknown")
        assert resp.status_code == 400
