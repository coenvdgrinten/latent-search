from typing import override
from unittest.mock import MagicMock, patch

from django.test import RequestFactory, TestCase
from latent_search.server.indexing.views.search_view import search_dashboard


class SearchDashboardViewTest(TestCase):
    @override
    def setUp(self):
        self.factory = RequestFactory()

    @patch("latent_search.server.indexing.views.search_view.render")
    @patch("latent_search.server.indexing.views.search_view.search_service")
    def test_get_without_query_renders_empty_results(self, mock_service, mock_render):
        """A GET request with no query string should render with empty results."""
        mock_render.return_value = MagicMock()

        request = self.factory.get("/")
        search_dashboard(request)

        mock_service.semantic_search.assert_not_called()
        _, template, context = mock_render.call_args[0]
        self.assertEqual(template, "indexing/dashboard.html")
        self.assertEqual(context["query"], "")
        self.assertEqual(context["results"], [])

    @patch("latent_search.server.indexing.views.search_view.render")
    @patch("latent_search.server.indexing.views.search_view.search_service")
    def test_get_with_query_calls_search_service(self, mock_service, mock_render):
        """GET with ?q= should call semantic_search and pass results to context."""
        mock_render.return_value = MagicMock()
        mock_service.semantic_search.return_value = [
            {"id": "1", "score": 0.9, "file_path": "/img.jpg", "file_name": "img.jpg"}
        ]

        request = self.factory.get("/", {"q": "sunset"})
        search_dashboard(request)

        mock_service.semantic_search.assert_called_once_with(query="sunset", limit=24)
        _, template, context = mock_render.call_args[0]
        self.assertEqual(template, "indexing/dashboard.html")
        self.assertEqual(context["query"], "sunset")
        self.assertEqual(len(context["results"]), 1)

    @patch("latent_search.server.indexing.views.search_view.render")
    @patch("latent_search.server.indexing.views.search_view.search_service")
    def test_get_with_whitespace_only_query_does_not_search(
        self, mock_service, mock_render
    ):
        """A query of only whitespace should be treated as no query."""
        mock_render.return_value = MagicMock()

        request = self.factory.get("/", {"q": "   "})
        search_dashboard(request)

        mock_service.semantic_search.assert_not_called()
        _, _, context = mock_render.call_args[0]
        self.assertEqual(context["query"], "")
        self.assertEqual(context["results"], [])
