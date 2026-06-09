"""
Integration tests for search ranking quality.

These tests run against the real Qdrant and CLIP instances to validate
that search results rank correctly for factual queries (locations, dates)
and visual/conceptual queries.

Run with:
    ./manage test latent_search.server.indexing.tests.test_search_integration

Requirements:
    - Qdrant running and reachable (QDRANT_URL env var or localhost:6333)
    - Collection indexed with media files
"""
from typing import override

from django.test import TestCase
from latent_search.server.indexing.services.search import (
    QdrantUnavailableError,
    SearchService,
)

# Test scenarios: (query, expected_top_result_filename_contains)
# These are queries that CURRENTLY FAIL with CLIP-text but SHOULD succeed.
SEARCH_SCENARIOS = [
    # --- Location queries ---
    ("photos from england", "england"),
    ("pictures from japan", "japan"),
    ("london", "england"),
    ("ireland", "irland"),
    ("trip to italy", "italy"),
    ("germany", "germany"),
    # --- Date queries ---
    ("pictures from 2012", "irland"),  # Only photo from 2012
    ("photos from 2016", "japan"),  # Only photo from 2016
    ("summer photos", "england"),  # England is Aug 2018 (summer)
    # --- Combined queries ---
    ("photos from england in 2018", "england"),
    ("trip to italy in 2018", "italy"),
    # --- Visual/conceptual queries ---
    ("water", ("italy", "japan")),  # Lake or river — both valid with text-only matching
    ("bridge", "england"),  # London bridge
]


class SearchRankingIntegrationTest(TestCase):
    """
    Integration tests that validate search ranking against real Qdrant + CLIP.

    Each scenario asserts that the expected photo ranks #1 for its query.
    """

    @classmethod
    @override
    def setUpClass(cls) -> None:
        super().setUpClass()
        cls.service = SearchService()

    def _rank_for_query(self, query: str) -> list[str]:
        """Run a search and return ordered list of filenames."""
        results = self.service.semantic_search(query, limit=5)
        return [r["file_name"] for r in results]

    def _assert_ranks_first(
        self,
        query: str,
        expected_filename_contains: str | tuple[str, ...],
    ) -> None:
        """Assert the expected photo ranks #1 for the given query.

        ``expected_filename_contains`` can be a single string or a tuple of
        acceptable substrings (e.g., ``("italy", "japan")`` for ambiguous
        visual queries).
        """
        try:
            ranks = self._rank_for_query(query)
        except QdrantUnavailableError:
            self.skipTest("Qdrant is not running or unreachable")

        self.assertTrue(
            len(ranks) > 0,
            f"No results returned for query '{query}'",
        )
        first_lower = ranks[0].lower()
        expected_values = (
            expected_filename_contains
            if isinstance(expected_filename_contains, tuple)
            else (expected_filename_contains,)
        )
        self.assertTrue(
            any(exp in first_lower for exp in expected_values),
            (
                f"Query '{query}': expected one of {expected_values} at #1, "
                f"got '{ranks[0]}'. Full ranking: {ranks}"
            ),
        )

    # --- Location queries ---

    def test_location_england(self):
        """'photos from england' should rank england-london-bridge.jpg first."""
        self._assert_ranks_first("photos from england", "england")

    def test_location_japan(self):
        """'pictures from japan' should rank japan-katsura-river.jpg first."""
        self._assert_ranks_first("pictures from japan", "japan")

    def test_location_london(self):
        """'london' should rank england-london-bridge.jpg first."""
        self._assert_ranks_first("london", "england")

    def test_location_ireland(self):
        """'ireland' should rank irland-dingle.jpg first."""
        self._assert_ranks_first("ireland", "irland")

    def test_location_trip_to_italy(self):
        """'trip to italy' should rank italy-garda-lake-sailing-club.jpg first."""
        self._assert_ranks_first("trip to italy", "italy")

    def test_location_germany(self):
        """'germany' should rank germany-english-garden.jpg first."""
        self._assert_ranks_first("germany", "germany")

    # --- Date queries ---

    def test_date_2012(self):
        """'pictures from 2012' should rank irland-dingle.jpg first."""
        self._assert_ranks_first("pictures from 2012", "irland")

    def test_date_2016(self):
        """'photos from 2016' should rank japan-katsura-river.jpg first."""
        self._assert_ranks_first("photos from 2016", "japan")

    def test_season_summer(self):
        """'summer photos' should rank england-london-bridge.jpg first (Aug 2018)."""
        self._assert_ranks_first("summer photos", "england")

    # --- Combined queries ---

    def test_combined_england_2018(self):
        """'photos from england in 2018' should rank england-london-bridge.jpg first."""
        self._assert_ranks_first("photos from england in 2018", "england")

    def test_combined_italy_2018(self):
        """'trip to italy in 2018' should rank italy-garda-lake first."""
        self._assert_ranks_first("trip to italy in 2018", "italy")

    # --- Visual/conceptual queries ---

    def test_visual_water(self):
        """'water' should rank a water-related photo first (lake or river)."""
        self._assert_ranks_first("water", ("italy", "japan"))

    def test_visual_bridge(self):
        """'bridge' should rank england-london-bridge.jpg first."""
        self._assert_ranks_first("bridge", "england")
