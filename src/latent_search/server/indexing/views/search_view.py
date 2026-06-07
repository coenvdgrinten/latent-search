from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from latent_search.server.indexing.services.search import QdrantUnavailableError, SearchService

search_service = SearchService()


def search_dashboard(request: HttpRequest) -> HttpResponse:
    """Renders the main semantic search interface and handles vector query execution."""
    query = request.GET.get("q", "").strip()
    results = []
    error = None

    if query:
        try:
            results = search_service.semantic_search(query=query, limit=24)
        except QdrantUnavailableError:
            error = "Search is unavailable: could not connect to the vector database. Is Qdrant running?"

    context = {
        "query": query,
        "results": results,
        "error": error,
    }
    return render(request, "indexing/dashboard.html", context)