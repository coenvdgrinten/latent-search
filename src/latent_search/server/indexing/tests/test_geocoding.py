"""Tests for the geocoding service."""

import os
import tempfile
from pathlib import Path
from typing import override
from unittest.mock import MagicMock, patch

from django.test import TestCase

from latent_search.server.indexing.services.geocoding import (
    USER_AGENT,
    GeocodingService,
)

_REQUESTS_GET_PATCH = "latent_search.server.indexing.services.geocoding.requests.get"


class UserAgentTest(TestCase):
    """Tests for the USER_AGENT constant."""

    def test_user_agent_contains_project_name(self):
        self.assertIn("LatentSearch", USER_AGENT)

    def test_user_agent_contains_contact_info(self):
        self.assertIn("github.com", USER_AGENT)


class GeocodingServiceInitTest(TestCase):
    """Tests for GeocodingService initialization."""

    def test_default_cache_db_location(self):
        with patch.object(GeocodingService, "_ensure_table"):
            service = GeocodingService()
            self.assertIn(".latent_search_geocache.sqlite", str(service.cache_db))

    def test_custom_cache_db(self):
        custom_path = Path("/tmp/custom_cache.sqlite")
        with patch.object(GeocodingService, "_ensure_table"):
            service = GeocodingService(cache_db=custom_path)
            self.assertEqual(service.cache_db, custom_path)

    def test_ensure_table_called_on_init(self):
        with patch.object(GeocodingService, "_ensure_table") as mock_ensure:
            GeocodingService()
            mock_ensure.assert_called_once()


class GeocodingServiceCacheKeyTest(TestCase):
    """Tests for the _cache_key static method."""

    def test_same_coords_produce_same_key(self):
        key1 = GeocodingService._cache_key(51.5074, -0.1278)
        key2 = GeocodingService._cache_key(51.5074, -0.1278)
        self.assertEqual(key1, key2)

    def test_different_coords_produce_different_keys(self):
        key1 = GeocodingService._cache_key(51.5074, -0.1278)
        key2 = GeocodingService._cache_key(48.8566, 2.3522)
        self.assertNotEqual(key1, key2)

    def test_swapped_coords_produce_different_keys(self):
        key1 = GeocodingService._cache_key(51.5074, -0.1278)
        key2 = GeocodingService._cache_key(-0.1278, 51.5074)
        self.assertNotEqual(key1, key2)

    def test_key_is_hex_digest(self):
        key = GeocodingService._cache_key(51.5074, -0.1278)
        self.assertEqual(len(key), 64)  # SHA-256 hex digest length
        # All chars should be hex
        int(key, 16)  # Will raise ValueError if not valid hex


class GeocodingServiceReverseGeocodeTest(TestCase):
    """Tests for the reverse_geocode method."""

    @override
    def setUp(self) -> None:
        fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        self.temp_db = Path(tmp_path)
        self.service = GeocodingService(cache_db=self.temp_db)

    @override
    def tearDown(self) -> None:
        if self.temp_db.exists():
            self.temp_db.unlink()

    def test_uncached_lookup_calls_nominatim(self):
        with patch(_REQUESTS_GET_PATCH) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "address": {
                    "suburb": "Southwark",
                    "city": "London",
                    "country": "United Kingdom",
                }
            }
            mock_get.return_value = mock_response

            self.service.reverse_geocode(51.5074, -0.1278)

            mock_get.assert_called_once()
            self.assertIn("nominatim.openstreetmap.org", mock_get.call_args[0][0])

    def test_cached_result_does_not_call_network(self):
        with patch(_REQUESTS_GET_PATCH) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "address": {
                    "suburb": "Southwark",
                    "city": "London",
                    "country": "United Kingdom",
                }
            }
            mock_get.return_value = mock_response

            self.service.reverse_geocode(51.5074, -0.1278)

            # Second call should use cache
            self.service.reverse_geocode(51.5074, -0.1278)

            mock_get.assert_called_once()

    def test_network_failure_returns_none(self):
        with patch(_REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            result = self.service.reverse_geocode(51.5074, -0.1278)

            self.assertIsNone(result)

    def test_negative_cache_for_failed_lookup(self):
        """Failed lookups are cached to avoid repeated network calls."""
        with patch(_REQUESTS_GET_PATCH) as mock_get:
            mock_get.side_effect = Exception("Connection refused")

            # First call — fails, caches None
            self.service.reverse_geocode(51.5074, -0.1278)

            # Patch succeeds on second attempt — but cache should prevent retry
            mock_get.side_effect = None
            mock_get.return_value = MagicMock(
                status_code=200,
                json=MagicMock(return_value={"display_name": "London"}),
            )

            # Second call — should use cached None, not retry network
            result = self.service.reverse_geocode(51.5074, -0.1278)

            self.assertIsNone(result)

    def test_different_coords_use_different_cache_entries(self):
        with patch(_REQUESTS_GET_PATCH) as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "address": {
                    "suburb": "Southwark",
                    "city": "London",
                    "country": "United Kingdom",
                }
            }
            mock_get.return_value = mock_response

            # Cache London
            self.service.reverse_geocode(51.5074, -0.1278)

            # Paris should miss cache
            self.service.reverse_geocode(48.8566, 2.3522)

            self.assertEqual(mock_get.call_count, 2)

    def test_ensure_table_creates_schema(self):
        """_ensure_table creates the geocache table if it doesn't exist."""
        fd, tmp_path = tempfile.mkstemp(suffix=".sqlite")
        os.close(fd)
        fresh_db = Path(tmp_path)
        try:
            GeocodingService(cache_db=fresh_db)
            # Table should exist now
            import sqlite3

            conn = sqlite3.connect(fresh_db)
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='geocache'"
            )
            row = cursor.fetchone()
            conn.close()
            self.assertIsNotNone(row)
        finally:
            if fresh_db.exists():
                fresh_db.unlink()
