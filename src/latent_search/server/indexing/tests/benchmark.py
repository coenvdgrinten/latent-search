"""
Benchmark infrastructure for measuring search quality.

Maintains a curated test corpus of queries with known-correct answers
based on our indexed media. Measures Recall@K and produces readable
reports suitable for comparing configurations (before/after changes).

Usage:
    from latent_search.server.indexing.tests.benchmark import (
        BenchmarkCollector,
        quick_eval,
    )
    from latent_search.server.indexing.services.search import SearchService

    service = SearchService()
    report = quick_eval(service.semantic_search)
    print(report)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Test corpus
# ---------------------------------------------------------------------------
# Each tuple: (query, expected_filename_substring, category)
#
# Ground truth derived from indexed media metadata:
#
#   england-london-bridge.jpg         London, Aug 2018 (summer)
#   germany-english-garden.jpg       Munich, Nov 2018 (autumn)
#   ireland-dingle.jpg               Dingle/Kerry, Sep 2012 (autumn)
#   italy-garda-lake-sailing-club.jpg Riva del Garda, Sep 2018 (autumn)
#   japan-katsura-river.jpg          Kyoto, Nov 2016 (autumn)
#
TEST_CORPUS: list[tuple[str, str, str]] = [
    # --- Location queries -----------------------------------------------
    ("photos from england", "england", "location"),
    ("pictures from japan", "japan", "location"),
    ("london", "england", "location"),
    ("ireland", "irland", "location"),  # filename uses Irish spelling
    ("trip to italy", "italy", "location"),
    ("germany", "germany", "location"),
    ("munich", "germany", "location"),
    ("dingle", "irland", "location"),  # filename is irland-dingle.jpg
    ("garda lake", "italy", "location"),
    # --- Year queries --------------------------------------------------
    ("pictures from 2012", "irland", "year"),  # only image from 2012
    ("photos from 2016", "japan", "year"),  # only image from 2016
    ("what happened in 2012", "irland", "year"),
    # --- Combined date + location --------------------------------------
    ("photos from england in 2018", "england", "combined"),
    ("trip to italy in 2018", "italy", "combined"),
    ("japan 2016", "japan", "combined"),
    ("ireland 2012", "irland", "combined"),  # filename uses Irish spelling
    # --- Season queries ------------------------------------------------
    ("summer photos", "england", "season"),  # only Aug image
]


class BenchmarkCollector:
    """
    Runs a set of test queries through a search function and measures
    Recall@K across categories.
    """

    def __init__(self) -> None:
        self.results: list[tuple[str, str, str, list[str]]] = []
        """Stored as (query, expected, category, ranked_filenames)."""

    # ---- Running -------------------------------------------------------

    def run(
        self,
        search_fn,
        *,
        corpus: list[tuple[str, str, str]] | None = None,
    ) -> None:
        """
        Execute all test queries through ``search_fn``.

        Args:
            search_fn: Callable ``(query: str, limit: int) -> list[dict]``
                       where each dict has a ``"file_name"`` key.
            corpus: Optional override for TEST_CORPUS.
        """
        items = corpus or TEST_CORPUS

        for query, expected, category in items:
            try:
                hits = search_fn(query, limit=10)
            except Exception:
                self.results.append((query, expected, category, []))
                continue

            filenames = [h["file_name"].lower() for h in hits]
            self.results.append((query, expected, category, filenames))

    # ---- Analysis -----------------------------------------------------

    def _hits_for_category(self, category: str) -> list[tuple[str, str, list[str]]]:
        return [
            (q, exp, ranks) for q, exp, cat, ranks in self.results if cat == category
        ]

    def categories(self) -> list[str]:
        return sorted({cat for _, _, cat, _ in self.results})

    def recall_at_k(self, k: int, category: str | None = None) -> float:
        """Fraction of queries where the expected answer appears in top-K."""
        items = (
            self._hits_for_category(category)
            if category
            else [(q, e, r) for q, e, _, r in self.results]
        )
        if not items:
            return 0.0

        hits = sum(1 for q, exp, ranks in items if any(exp in r for r in ranks[:k]))
        return hits / len(items)

    def missed_queries(self, k: int = 1) -> list[tuple[str, str, str, list[str]]]:
        """Return (query, expected, got_first, ranks) for failures at top-K."""
        missed: list[tuple[str, str, str, list[str]]] = []
        for query, expected, _category, ranks in self.results:
            if not any(expected in r for r in ranks[:k]):
                first = ranks[0] if ranks else "(no results)"
                missed.append((query, expected, first, ranks))
        return missed

    # ---- Reporting ----------------------------------------------------

    def report(self, k_values: list[int] | None = None) -> str:
        """Printable summary with per-category Recall@K and missed queries."""
        if not k_values:
            k_values = [1, 2, 3]

        lines: list[str] = []
        lines.append("=== Search Quality Report ===")
        lines.append("")

        # Per-category table
        cats = self.categories()
        header = f"{'Category':<15}" + "".join(f"  R @{v:>2}" for v in k_values)
        lines.append(header)
        lines.append("-" * len(header))

        for cat in cats:
            row = f"{cat:<15}"
            for kv in k_values:
                r = self.recall_at_k(kv, cat)
                row += f"  {r:>5.1%}"
            lines.append(row)

        # Overall totals
        overall = f"{'OVERALL':<15}"
        for kv in k_values:
            r = self.recall_at_k(kv)
            overall += f"  {r:>5.1%}"
        lines.append("-" * len(header))
        lines.append(overall)
        lines.append(f"Queries: {len(self.results)}")
        lines.append("")

        # Missed queries at R@1
        missed = self.missed_queries(1)
        if missed:
            lines.append("Missed at R@1:")
            for query, expected, got, _ in missed:
                lines.append(f"  '{query}' → expected '{expected}', got '{got}'")
            lines.append("")

        return "\n".join(lines)


def quick_eval(search_fn) -> str:
    """One-shot convenience: run corpus through search_fn, return report."""
    collector = BenchmarkCollector()
    collector.run(search_fn)
    return collector.report()
