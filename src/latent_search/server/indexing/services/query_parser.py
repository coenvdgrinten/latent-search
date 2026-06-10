"""Lightweight query parser for extracting structured entities.

Uses regex-based extraction rather than NER models to keep dependencies
lightweight and inference fast. Targets the most impactful entity types
for photo search: years, seasons, months, and location keywords.
"""

import re
from dataclasses import dataclass

MONTH_MAP = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "sep": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

SEASON_MONTHS: dict[str, set[int]] = {
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "autumn": {9, 10, 11},
    "fall": {9, 10, 11},
    "winter": {12, 1, 2},
}

FILLER_WORDS = frozenset(
    [
        # Common query prefixes/phrases
        "photos",
        "photo",
        "pictures",
        "picture",
        "images",
        "image",
        "pics",
        "pic",
        "from",
        "of",
        "my",
        "the",
        "in",
        "on",
        "at",
        "to",
        "for",
        "during",
        "around",
        "about",
        "trip",
        "vacation",
        "holiday",
    ]
)


@dataclass
class ParsedQuery:
    """Structured entities extracted from a natural language query."""

    year: int | None = None
    month: int | None = None
    season_months: set[int] | None = None
    location_keyword: str | None = None
    semantic_query: str = ""


def parse_query(query: str) -> ParsedQuery:
    """
    Extract structured entities from a natural language query.

    Returns a ParsedQuery with any detected entities and the remaining
    semantic query text (with extracted entities removed).

    Examples:
        >>> pq = parse_query("photos from england in 2012")
        >>> pq.year  # 2012
        >>> pq.location_keyword  # "england"
        >>> pq.semantic_query  # "photos from"
    """
    original = query
    remaining = query.lower().strip()

    result = ParsedQuery()

    # --- Year detection (4-digit year) ---
    year_match = re.search(r"\b(19\d{2}|20\d{2})\b", remaining)
    if year_match:
        result.year = int(year_match.group(1))
        remaining = remaining[: year_match.start()] + remaining[year_match.end() :]

    # --- Month detection ---
    for month_name, month_num in MONTH_MAP.items():
        if re.search(rf"\b{month_name}\b", remaining):
            result.month = month_num
            remaining = re.sub(rf"\b{month_name}\b", "", remaining, flags=re.IGNORECASE)
            break

    # --- Season detection ---
    for season, months in SEASON_MONTHS.items():
        if re.search(rf"\b{season}\b", remaining):
            result.season_months = months
            remaining = re.sub(rf"\b{season}\b", "", remaining, flags=re.IGNORECASE)
            break

    # --- Location keyword extraction ---
    # Strip common filler words/phrases, leaving location-like terms
    cleaned = re.sub(
        rf"\b({'|'.join(FILLER_WORDS)})\b",
        "",
        remaining,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if cleaned and not re.match(r"^[\d\s\-\.]+$", cleaned):
        result.location_keyword = cleaned

    # Semantic query is whatever remains after stripping entities
    result.semantic_query = original

    return result
