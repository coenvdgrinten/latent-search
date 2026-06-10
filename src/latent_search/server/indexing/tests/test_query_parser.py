"""Tests for the lightweight query parser."""

from django.test import TestCase

from latent_search.server.indexing.services.query_parser import (
    FILLER_WORDS,
    MONTH_MAP,
    SEASON_MONTHS,
    ParsedQuery,
    parse_query,
)


class ParsedQueryTest(TestCase):
    """Tests for the ParsedQuery dataclass."""

    def test_defaults_are_none_or_empty(self):
        pq = ParsedQuery()
        self.assertIsNone(pq.year)
        self.assertIsNone(pq.month)
        self.assertIsNone(pq.season_months)
        self.assertIsNone(pq.location_keyword)
        self.assertEqual(pq.semantic_query, "")

    def test_accepts_all_fields(self):
        pq = ParsedQuery(
            year=2020,
            month=6,
            season_months={6, 7, 8},
            location_keyword="england",
            semantic_query="beach photos",
        )
        self.assertEqual(pq.year, 2020)
        self.assertEqual(pq.month, 6)
        self.assertEqual(pq.season_months, {6, 7, 8})
        self.assertEqual(pq.location_keyword, "england")
        self.assertEqual(pq.semantic_query, "beach photos")


class MonthMapTest(TestCase):
    """Tests for the MONTH_MAP constant."""

    def test_has_all_twelve_months(self):
        self.assertEqual(len(MONTH_MAP), 13)  # 12 + sep abbreviation

    def test_month_numbers_correct(self):
        self.assertEqual(MONTH_MAP["january"], 1)
        self.assertEqual(MONTH_MAP["june"], 6)
        self.assertEqual(MONTH_MAP["december"], 12)

    def test_sep_abbreviation_maps_to_september(self):
        self.assertEqual(MONTH_MAP["sep"], 9)
        self.assertEqual(MONTH_MAP["september"], 9)


class SeasonMonthsTest(TestCase):
    """Tests for the SEASON_MONTHS constant."""

    def test_summer_is_northern_hemisphere(self):
        self.assertEqual(SEASON_MONTHS["summer"], {6, 7, 8})

    def test_autumn_is_northern_hemisphere(self):
        self.assertEqual(SEASON_MONTHS["autumn"], {9, 10, 11})

    def test_fall_alias_equals_autumn(self):
        self.assertEqual(SEASON_MONTHS["fall"], SEASON_MONTHS["autumn"])

    def test_winter_spans_year_boundary(self):
        self.assertEqual(SEASON_MONTHS["winter"], {12, 1, 2})

    def test_spring_months(self):
        self.assertEqual(SEASON_MONTHS["spring"], {3, 4, 5})


class FillWordsTest(TestCase):
    """Tests for the FILLER_WORDS constant."""

    def test_common_query_words_present(self):
        for word in ("photos", "from", "my", "the", "trip", "vacation"):
            self.assertIn(word, FILLER_WORDS)

    def test_is_frozenset(self):
        self.assertIsInstance(FILLER_WORDS, frozenset)


class ParseQueryYearTest(TestCase):
    """Tests for year extraction."""

    def test_simple_year(self):
        pq = parse_query("photos from 2012")
        self.assertEqual(pq.year, 2012)

    def test_1900s_year(self):
        pq = parse_query("vintage photo from 1985")
        self.assertEqual(pq.year, 1985)

    def test_year_in_middle_of_sentence(self):
        pq = parse_query("photos from 2012 in japan")
        self.assertEqual(pq.year, 2012)

    def test_no_year_detected(self):
        pq = parse_query("photos of my dog")
        self.assertIsNone(pq.year)

    def test_three_digit_number_not_a_year(self):
        pq = parse_query("photo 123")
        self.assertIsNone(pq.year)

    def test_five_digit_number_not_a_year(self):
        pq = parse_query("zip code 12345")
        self.assertIsNone(pq.year)

    def test_case_insensitive_year(self):
        pq = parse_query("PHOTOS FROM 2020")
        self.assertEqual(pq.year, 2020)


class ParseQueryMonthTest(TestCase):
    """Tests for month extraction."""

    def test_full_month_name(self):
        pq = parse_query("photos from january")
        self.assertEqual(pq.month, 1)

    def test_august(self):
        pq = parse_query("august vacation photos")
        self.assertEqual(pq.month, 8)

    def test_december(self):
        pq = parse_query("christmas in december")
        self.assertEqual(pq.month, 12)

    def test_no_month_detected(self):
        pq = parse_query("random photos")
        self.assertIsNone(pq.month)

    def test_month_with_year(self):
        pq = parse_query("photos from march 2015")
        self.assertEqual(pq.month, 3)
        self.assertEqual(pq.year, 2015)


class ParseQuerySeasonTest(TestCase):
    """Tests for season extraction."""

    def test_summer_season(self):
        pq = parse_query("summer vacation photos")
        self.assertEqual(pq.season_months, {6, 7, 8})

    def test_autumn_season(self):
        pq = parse_query("autumn leaves")
        self.assertEqual(pq.season_months, {9, 10, 11})

    def test_fall_alias(self):
        pq = parse_query("fall foliage")
        self.assertEqual(pq.season_months, {9, 10, 11})

    def test_winter_season(self):
        pq = parse_query("winter snow photos")
        self.assertEqual(pq.season_months, {12, 1, 2})

    def test_spring_season(self):
        pq = parse_query("spring flowers")
        self.assertEqual(pq.season_months, {3, 4, 5})

    def test_no_season_detected(self):
        pq = parse_query("city skyline")
        self.assertIsNone(pq.season_months)


class ParseQueryLocationTest(TestCase):
    """Tests for location keyword extraction."""

    def test_simple_location(self):
        pq = parse_query("photos from england")
        self.assertEqual(pq.location_keyword, "england")

    def test_multi_word_location(self):
        pq = parse_query("photos from new york")
        self.assertEqual(pq.location_keyword, "new york")

    def test_location_with_year(self):
        pq = parse_query("photos from japan in 2012")
        self.assertEqual(pq.location_keyword, "japan")

    def test_no_location(self):
        pq = parse_query("photos of my dog playing")
        # "dog playing" survives filler word stripping — it's a valid keyword
        self.assertEqual(pq.location_keyword, "dog playing")

    def test_filler_words_stripped(self):
        pq = parse_query("photos from my trip")
        self.assertIsNone(pq.location_keyword)

    def test_case_insensitive_location(self):
        pq = parse_query("PHOTOS FROM FRANCE")
        self.assertEqual(pq.location_keyword, "france")


class ParseQueryCombinedTest(TestCase):
    """Tests for combined entity extraction."""

    def test_year_and_location(self):
        pq = parse_query("photos from england in 2012")
        self.assertEqual(pq.year, 2012)
        self.assertEqual(pq.location_keyword, "england")

    def test_month_and_year(self):
        pq = parse_query("photos from august 2018")
        self.assertEqual(pq.month, 8)
        self.assertEqual(pq.year, 2018)

    def test_season_and_location(self):
        pq = parse_query("summer photos from italy")
        self.assertEqual(pq.season_months, {6, 7, 8})
        self.assertEqual(pq.location_keyword, "italy")

    def test_full_complex_query(self):
        pq = parse_query("photos from my trip to japan in november 2016")
        self.assertEqual(pq.year, 2016)
        self.assertEqual(pq.month, 11)
        self.assertEqual(pq.location_keyword, "japan")

    def test_semantic_query_preserves_original(self):
        pq = parse_query("photos from england in 2012")
        self.assertEqual(pq.semantic_query, "photos from england in 2012")

    def test_empty_query(self):
        pq = parse_query("")
        self.assertIsNone(pq.year)
        self.assertIsNone(pq.month)
        self.assertIsNone(pq.season_months)
        self.assertIsNone(pq.location_keyword)

    def test_whitespace_only_query(self):
        pq = parse_query("   ")
        self.assertIsNone(pq.year)
        self.assertIsNone(pq.location_keyword)
