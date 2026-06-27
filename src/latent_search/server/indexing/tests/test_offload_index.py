"""Tests for the offload_index management command.

Validates the cross-machine indexing pipeline wiring with mocked HTTP
transport and mocked ML services — no GPU, no network, no real Qdrant.
"""

import json
from unittest import mock
from unittest.mock import patch

from django.core.management import call_command
from django.test import TestCase

# Patch paths — where each service class is imported inside the command.
_CMD_MOD = "latent_search.server.indexing.management.commands.offload_index"
_CLIP_PATCH = f"{_CMD_MOD}.CLIPService"
_VLM_PATCH = f"{_CMD_MOD}.VLMService"
_TEXT_PATCH = f"{_CMD_MOD}.TextEmbeddingService"
_SPARSE_PATCH = f"{_CMD_MOD}.SparseEncodingService"
_GEO_PATCH = f"{_CMD_MOD}.GeocodingService"
_VDB_PATCH = f"{_CMD_MOD}.VectorDBService"
_REQUESTS_SESSION_PATCH = f"{_CMD_MOD}.requests.Session"


def _make_export_response(records: list[dict]):
    """Build a fake requests.Response that streams NDJSON lines."""
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.raise_for_status = mock.MagicMock()
    lines = [b"# header"] + [json.dumps(r).encode("utf-8") for r in records]
    resp.iter_lines = mock.MagicMock(return_value=lines)
    return resp


def _make_image_response(content: bytes = b"\xff\xd8\xff\xe0fakejpegdata"):
    """Build a fake requests.Response for an image download."""
    resp = mock.MagicMock()
    resp.status_code = 200
    resp.iter_content = mock.MagicMock(return_value=[content])
    return resp


def _make_login_response(status: int = 302):
    resp = mock.MagicMock()
    resp.status_code = status
    resp.text = ""
    return resp


def _make_stats_response(status: int = 200):
    resp = mock.MagicMock()
    resp.status_code = status
    return resp


def _make_import_response(status: int = 200, body: dict | None = None):
    resp = mock.MagicMock()
    resp.status_code = status
    resp.json = mock.MagicMock(return_value=body or {"created": 0, "updated": 1})
    resp.text = json.dumps(body or {})
    return resp


class OffloadIndexCommandTest(TestCase):
    """Wiring tests for the offload_index command."""

    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_SPARSE_PATCH)
    @patch(_TEXT_PATCH)
    @patch(_VLM_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_REQUESTS_SESSION_PATCH)
    def test_full_pipeline_wiring(
        self,
        mock_session_class,
        mock_clip_class,
        mock_vlm_class,
        mock_text_class,
        mock_sparse_class,
        mock_geo_class,
        mock_vdb_class,
    ):
        """End-to-end wiring: fetch → image → VLM → CLIP → BGE → Qdrant → writeback."""
        # ── Mock ML services ──
        mock_clip = mock_clip_class.return_value
        mock_clip.get_image_embedding.return_value = [0.1] * 1024
        mock_vlm = mock_vlm_class.return_value
        mock_vlm.describe.return_value = "A red car on a road"
        mock_text = mock_text_class.return_value
        mock_text.encode.return_value = [0.2] * 1024
        mock_sparse = mock_sparse_class.return_value
        mock_sparse.encode_document.return_value = {
            "indices": [1, 2],
            "values": [0.5, 0.4],
        }
        mock_geo = mock_geo_class.return_value
        mock_geo.reverse_geocode.return_value = "Amsterdam"
        mock_vdb = mock_vdb_class.return_value

        # ── Mock HTTP session ──
        session = mock.MagicMock()
        mock_session_class.return_value = session

        record = {
            "id": 1,
            "file_path": "/nc_data/photos/car.jpg",
            "filename": "car.jpg",
            "relative_path": "photos/car.jpg",
            "file_size": 1000,
            "mime_type": "image/jpeg",
            "taken_at": "2024-07-15T10:30:00",
            "latitude": 52.0,
            "longitude": 5.0,
            "width": 1920,
            "height": 1080,
            "vlm_caption": "",
            "vector_id": None,
            "is_indexed": False,
        }

        # Configure session.get to handle login probe, export, and image fetch.
        def get_side_effect(url, **kwargs):
            if url.endswith("/api/stats"):
                return _make_stats_response(200)
            if url.endswith("/api/export_library"):
                return _make_export_response([record])
            if "/api/media/1/image" in url:
                return _make_image_response()
            return mock.MagicMock(status_code=404)

        session.get.side_effect = get_side_effect
        session.post.side_effect = lambda url, **kwargs: (
            _make_login_response(302)
            if url.endswith("/login/")
            else _make_import_response(200, {"created": 0, "updated": 1})
        )

        # ── Run ──
        call_command(
            "offload_index",
            "--api-url",
            "http://unraid:8000",
            "--api-user",
            "admin",
            "--api-password",
            "secret",
            "--qdrant-url",
            "http://unraid:6333",
        )

        # ── Assertions ──
        # VLM was called (caption was empty)
        mock_vlm.describe.assert_called_once()
        # CLIP + BGE + sparse each called once
        mock_clip.get_image_embedding.assert_called_once()
        mock_text.encode.assert_called_once()
        mock_sparse.encode_document.assert_called_once()
        # Qdrant upsert happened
        mock_vdb.ensure_collection.assert_called_once()
        mock_vdb.upsert_embedding.assert_called_once()
        upsert_kwargs = mock_vdb.upsert_embedding.call_args.kwargs
        self.assertEqual(len(upsert_kwargs["image_embedding"]), 1024)
        self.assertEqual(len(upsert_kwargs["text_embedding"]), 1024)
        self.assertIn("caption", upsert_kwargs["payload"])
        # Writeback POST happened
        post_calls = session.post.call_args_list
        # First call is login, second is import_library
        self.assertEqual(len(post_calls), 2)
        import_call = post_calls[1]
        self.assertIn("/api/import_library", import_call.args[0])
        # The uploaded NDJSON should contain the caption and is_indexed=True
        files_kw = import_call.kwargs.get("files", {})
        uploaded_bytes = files_kw["file"][1]
        if isinstance(uploaded_bytes, bytes):
            uploaded_text = uploaded_bytes.decode("utf-8")
        else:
            uploaded_text = uploaded_bytes.read().decode("utf-8")
        writeback_records = [
            json.loads(line)
            for line in uploaded_text.splitlines()
            if line and not line.startswith("#")
        ]
        self.assertEqual(len(writeback_records), 1)
        self.assertEqual(writeback_records[0]["vlm_caption"], "A red car on a road")
        self.assertTrue(writeback_records[0]["is_indexed"])

    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_SPARSE_PATCH)
    @patch(_TEXT_PATCH)
    @patch(_VLM_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_REQUESTS_SESSION_PATCH)
    def test_dry_run_skips_qdrant_and_writeback(
        self,
        mock_session_class,
        mock_clip_class,
        mock_vlm_class,
        mock_text_class,
        mock_sparse_class,
        mock_geo_class,
        mock_vdb_class,
    ):
        """--dry-run should do ML work but skip Qdrant upsert and import POST."""
        mock_clip_class.return_value.get_image_embedding.return_value = [0.0] * 1024
        mock_vlm_class.return_value.describe.return_value = "caption"
        mock_text_class.return_value.encode.return_value = [0.0] * 1024
        mock_sparse_class.return_value.encode_document.return_value = {
            "indices": [],
            "values": [],
        }

        session = mock.MagicMock()
        mock_session_class.return_value = session
        record = {
            "id": 1,
            "file_path": "/x.jpg",
            "filename": "x.jpg",
            "relative_path": "x.jpg",
            "file_size": 1,
            "vlm_caption": "",
            "vector_id": None,
            "is_indexed": False,
        }

        def get_side_effect(url, **kwargs):
            if url.endswith("/api/stats"):
                return _make_stats_response(200)
            if url.endswith("/api/export_library"):
                return _make_export_response([record])
            if "/api/media/1/image" in url:
                return _make_image_response()
            return mock.MagicMock(status_code=404)

        session.get.side_effect = get_side_effect

        def post_side_effect(url, **kwargs):
            if url.endswith("/login/"):
                return _make_login_response(302)
            return _make_import_response()

        session.post.side_effect = post_side_effect

        call_command(
            "offload_index",
            "--api-url",
            "http://unraid:8000",
            "--api-user",
            "admin",
            "--api-password",
            "secret",
            "--dry-run",
        )

        mock_vdb_class.return_value.ensure_collection.assert_not_called()
        mock_vdb_class.return_value.upsert_embedding.assert_not_called()
        # Only the login POST should have happened, no import_library POST
        post_calls = session.post.call_args_list
        self.assertEqual(len(post_calls), 1)
        self.assertTrue(post_calls[0].args[0].endswith("/login/"))

    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_SPARSE_PATCH)
    @patch(_TEXT_PATCH)
    @patch(_VLM_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_REQUESTS_SESSION_PATCH)
    def test_skip_vlm_uses_existing_caption(
        self,
        mock_session_class,
        mock_clip_class,
        mock_vlm_class,
        mock_text_class,
        mock_sparse_class,
        mock_geo_class,
        mock_vdb_class,
    ):
        """--skip-vlm should not instantiate VLM and should use the exported caption."""
        mock_clip_class.return_value.get_image_embedding.return_value = [0.0] * 1024
        mock_text_class.return_value.encode.return_value = [0.0] * 1024
        mock_sparse_class.return_value.encode_document.return_value = {
            "indices": [],
            "values": [],
        }

        session = mock.MagicMock()
        mock_session_class.return_value = session
        record = {
            "id": 1,
            "file_path": "/x.jpg",
            "filename": "x.jpg",
            "relative_path": "x.jpg",
            "file_size": 1,
            "vlm_caption": "pre-existing caption",
            "vector_id": None,
            "is_indexed": False,
        }

        def get_side_effect(url, **kwargs):
            if url.endswith("/api/stats"):
                return _make_stats_response(200)
            if url.endswith("/api/export_library"):
                return _make_export_response([record])
            if "/api/media/1/image" in url:
                return _make_image_response()
            return mock.MagicMock(status_code=404)

        session.get.side_effect = get_side_effect

        def post_side_effect(url, **kwargs):
            if url.endswith("/login/"):
                return _make_login_response(302)
            return _make_import_response()

        session.post.side_effect = post_side_effect

        call_command(
            "offload_index",
            "--api-url",
            "http://unraid:8000",
            "--api-user",
            "admin",
            "--api-password",
            "secret",
            "--skip-vlm",
        )

        mock_vlm_class.assert_not_called()
        # The text embedding should have been built from the existing caption
        encode_arg = mock_text_class.return_value.encode.call_args.args[0]
        self.assertIn("pre-existing caption", encode_arg)

    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_SPARSE_PATCH)
    @patch(_TEXT_PATCH)
    @patch(_VLM_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_REQUESTS_SESSION_PATCH)
    def test_login_failure_raises_command_error(
        self,
        mock_session_class,
        mock_clip_class,
        mock_vlm_class,
        mock_text_class,
        mock_sparse_class,
        mock_geo_class,
        mock_vdb_class,
    ):
        """A failed login probe should abort the command."""
        from django.core.management import CommandError

        session = mock.MagicMock()
        mock_session_class.return_value = session
        session.post.side_effect = lambda url, **kwargs: _make_login_response(302)
        # Probe returns 401 → not authenticated
        session.get.side_effect = lambda url, **kwargs: _make_stats_response(401)

        with self.assertRaises(CommandError):
            call_command(
                "offload_index",
                "--api-url",
                "http://unraid:8000",
                "--api-user",
                "admin",
                "--api-password",
                "wrong",
            )

    @patch(_VDB_PATCH)
    @patch(_GEO_PATCH)
    @patch(_SPARSE_PATCH)
    @patch(_TEXT_PATCH)
    @patch(_VLM_PATCH)
    @patch(_CLIP_PATCH)
    @patch(_REQUESTS_SESSION_PATCH)
    def test_no_pending_records_exits_cleanly(
        self,
        mock_session_class,
        mock_clip_class,
        mock_vlm_class,
        mock_text_class,
        mock_sparse_class,
        mock_geo_class,
        mock_vdb_class,
    ):
        """An empty export should exit without touching ML services."""
        session = mock.MagicMock()
        mock_session_class.return_value = session
        session.post.side_effect = lambda url, **kwargs: _make_login_response(302)

        def get_side_effect(url, **kwargs):
            if url.endswith("/api/stats"):
                return _make_stats_response(200)
            if url.endswith("/api/export_library"):
                return _make_export_response([])
            return mock.MagicMock(status_code=404)

        session.get.side_effect = get_side_effect

        call_command(
            "offload_index",
            "--api-url",
            "http://unraid:8000",
            "--api-user",
            "admin",
            "--api-password",
            "secret",
        )

        mock_clip_class.return_value.get_image_embedding.assert_not_called()
        mock_vdb_class.return_value.upsert_embedding.assert_not_called()
