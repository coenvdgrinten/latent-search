import hashlib
import logging
import sqlite3
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

USER_AGENT = "LatentSearch/1.0 (https://github.com/coenvdgrinten/latent-search)"


class GeocodingService:
    """
    Reverse-geocode latitude/longitude pairs into human-readable location
    hierarchies using the OpenStreetMap Nominatim API.

    Results are cached in a local SQLite database keyed by a hash of the
    coordinate pair, so repeated lookups (or re-indexing) don't hit the
    network.  Requests are throttled to ~1 req/sec to respect Nominatim's
    fair-use policy.
    """

    CACHE_DB = Path.home() / ".latent_search_geocache.sqlite"

    def __init__(self, cache_db: Path | None = None):
        self.cache_db = cache_db or self.CACHE_DB
        self._lock = threading.Lock()
        self._last_request_time = 0.0
        self._ensure_table()

    def reverse_geocode(self, latitude: float, longitude: float) -> str | None:
        """
        Return a comma-separated location string
        (e.g. ``"Southwark, London, England, United Kingdom"``) or ``None``
        if the lookup fails or is unavailable.
        """
        cache_key = self._cache_key(latitude, longitude)

        # Fast-path: check cache
        cached = self._get_cached(cache_key)
        if cached is not None:
            return cached

        # Slow-path: hit Nominatim
        try:
            result = self._lookup(latitude, longitude)
        except Exception as e:
            logger.debug(f"Nominatim lookup failed for ({latitude},{longitude}): {e}")
            return None

        # Cache whatever we got (even None counts as a negative cache)
        self._store_cache(cache_key, result)
        return result

    @staticmethod
    def _cache_key(lat: float, lon: float) -> str:
        return hashlib.sha256(f"{lat:.6f},{lon:.6f}".encode()).hexdigest()

    def _ensure_table(self) -> None:
        conn = sqlite3.connect(self.cache_db)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS geocache (
                cache_key TEXT PRIMARY KEY,
                location TEXT
            )
        """)
        conn.commit()
        conn.close()

    def _get_cached(self, cache_key: str) -> str | None:
        conn = sqlite3.connect(self.cache_db)
        cursor = conn.execute(
            "SELECT location FROM geocache WHERE cache_key = ?", (cache_key,)
        )
        row = cursor.fetchone()
        conn.close()
        return row[0] if row else None

    def _store_cache(self, cache_key: str, location: str | None) -> None:
        conn = sqlite3.connect(self.cache_db)
        conn.execute(
            "INSERT OR REPLACE INTO geocache (cache_key, location) VALUES (?, ?)",
            (cache_key, location),
        )
        conn.commit()
        conn.close()

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_time
        if elapsed < 1.1:  # ~1 req/sec + margin
            time.sleep(1.1 - elapsed)

    def _lookup(self, lat: float, lon: float) -> str | None:
        self._throttle()
        self._last_request_time = time.monotonic()

        resp = requests.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={
                "lat": lat,
                "lon": lon,
                "format": "json",
                "zoom": 18,
                "addressdetails": 1,
            },
            headers={"User-Agent": USER_AGENT},
            timeout=10,
        )
        resp.raise_for_status()
        address = resp.json().get("address", {})

        parts = []
        for key in (
            "neighbourhood",
            "suburb",
            "city",
            "town",
            "village",
            "county",
            "state",
            "country",
        ):
            if address.get(key):
                parts.append(address[key])

        return ", ".join(parts) if parts else None
