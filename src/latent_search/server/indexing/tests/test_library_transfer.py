"""Tests for library export/import transfer workflow."""

import json
import uuid
from datetime import datetime
from typing import override

from django.contrib.auth.models import User
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import Client, TestCase


class ExportImportTest(TestCase):
    """Verify NDJSON export and upsert-import round-trip."""

    @override
    def setUp(self) -> None:
        self.client = Client()
        self.user = User.objects.create_user(username="admin", password="pass")
        self.client.login(username="admin", password="pass")
        self.id1 = uuid.uuid4()
        self.id2 = uuid.uuid4()
        self.taken = datetime(2024, 7, 15, 10, 30, 0)

        # Three records with varying caption states
        from latent_search.server.indexing.models.media import IndexedMedia

        IndexedMedia.objects.bulk_create(
            [
                IndexedMedia(
                    file_path="/photos/a.jpg",
                    filename="a.jpg",
                    relative_path="a.jpg",
                    file_size=1000,
                    mime_type="image/jpeg",
                    is_indexed=True,
                    vector_id=self.id1,
                    taken_at=self.taken,
                    latitude=52.0,
                    longitude=5.0,
                    width=1920,
                    height=1080,
                    caption="sample caption A",
                    vlm_caption="A red sunset over mountains",
                ),
                IndexedMedia(
                    file_path="/photos/b.jpg",
                    filename="b.jpg",
                    relative_path="b.jpg",
                    file_size=2000,
                    mime_type="image/jpeg",
                    is_indexed=True,
                    vector_id=self.id2,
                    vlm_caption="",  # uncaptioned
                ),
                IndexedMedia(
                    file_path="/photos/c.raw",
                    filename="c.raw",
                    relative_path="c.raw",
                    file_size=5000,
                    mime_type="image/x-canon-cr2",
                    is_indexed=False,  # not indexed yet
                    vlm_caption="",
                ),
            ]
        )

    def _parse_export_response(self, resp) -> list[dict]:
        """Extract JSON records from an export response."""
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp["Content-Type"], "application/x-ndjson")
        full_text = "".join(chunk.decode() for chunk in resp.streaming_content)
        records: list[dict] = []
        for line in full_text.splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            records.append(json.loads(stripped))
        return records

    # ── Export Tests ──

    def test_export_all_returns_two_indexed_records(self) -> None:
        resp = self.client.get(
            "/api/export_library?indexed_only=true&uncaptioned_only=false"
        )
        records = self._parse_export_response(resp)
        # a.jpg and b.jpg are indexed; c.raw is not
        self.assertEqual(len(records), 2)

    def test_export_no_filter_returns_all_records(self) -> None:
        resp = self.client.get("/api/export_library")
        records = self._parse_export_response(resp)
        self.assertEqual(len(records), 3)

    def test_export_uncaptioned_filters_captions(self) -> None:
        resp = self.client.get(
            "/api/export_library?indexed_only=true&uncaptioned_only=true"
        )
        records = self._parse_export_response(resp)
        paths = {r["file_path"] for r in records}
        # Only b.jpg is indexed AND uncaptioned
        self.assertEqual(paths, {"/photos/b.jpg"})

    def test_export_contains_schema_version(self) -> None:
        resp = self.client.get("/api/export_library")
        full_text = "".join(chunk.decode() for chunk in resp.streaming_content)
        first_line = full_text.splitlines()[0].strip()
        self.assertTrue(first_line.startswith("#"))
        self.assertIn("schema", first_line.lower())

    def test_export_serializes_datetime_and_uuid(self) -> None:
        resp = self.client.get("/api/export_library?indexed_only=true")
        records = self._parse_export_response(resp)
        a_rec = next(r for r in records if r["file_path"] == "/photos/a.jpg")
        self.assertIsInstance(a_rec["taken_at"], str)
        self.assertEqual(a_rec["vector_id"], str(self.id1))

    # ── Import Tests ──

    def _upload_ndjson(self, lines: list[str]) -> dict:
        """Helper: upload NDJSON content and return parsed JSON response."""
        blob = "\n".join(lines).encode("utf-8")
        uploaded = SimpleUploadedFile(
            "test.ndjson", blob, content_type="application/x-ndjson"
        )
        resp = self.client.post(
            "/api/import_library",
            {"file": uploaded},
        )
        self.assertEqual(resp.status_code, 200)
        return json.loads(resp.content)

    def test_import_creates_new_records(self) -> None:
        new_record = {
            "version": 1,
            "file_path": "/photos/new.jpg",
            "filename": "new.jpg",
            "relative_path": "new.jpg",
            "file_size": 999,
            "mime_type": "image/jpeg",
            "is_indexed": False,
            "vlm_caption": "",
        }
        result = self._upload_ndjson([json.dumps(new_record)])
        self.assertEqual(result["created"], 1)
        self.assertEqual(result["updated"], 0)

        from latent_search.server.indexing.models.media import IndexedMedia

        self.assertTrue(
            IndexedMedia.objects.filter(file_path="/photos/new.jpg").exists()
        )

    def test_import_updates_existing_records(self) -> None:
        # Simulate receiving an updated caption for /photos/a.jpg
        update_record = {
            "version": 1,
            "file_path": "/photos/a.jpg",
            "vlm_caption": "Updated GPU-generated caption",
        }
        result = self._upload_ndjson([json.dumps(update_record)])
        self.assertEqual(result["created"], 0)
        self.assertEqual(result["updated"], 1)

        from latent_search.server.indexing.models.media import IndexedMedia

        obj = IndexedMedia.objects.get(file_path="/photos/a.jpg")
        self.assertEqual(obj.vlm_caption, "Updated GPU-generated caption")

    def test_import_skips_comments_and_blank_lines(self) -> None:
        record = {
            "version": 1,
            "file_path": "/photos/d.png",
            "filename": "d.png",
            "relative_path": "d.png",
            "file_size": 100,
            "mime_type": "image/png",
            "is_indexed": False,
            "vlm_caption": "",
        }
        result = self._upload_ndjson(
            [
                "# comment line",
                "",
                json.dumps(record),
                "   ",
            ]
        )
        self.assertEqual(result["total_lines"], 1)
        self.assertEqual(result["created"], 1)

    def test_import_method_guard(self) -> None:
        resp = self.client.get("/api/import_library")
        self.assertEqual(resp.status_code, 405)

    def test_import_missing_file(self) -> None:
        resp = self.client.post("/api/import_library")
        self.assertEqual(resp.status_code, 400)
        data = json.loads(resp.content)
        self.assertIn("No file", data["error"])
