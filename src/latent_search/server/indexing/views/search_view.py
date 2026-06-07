from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from latent_search.server.indexing.services.search import SearchService

search_service = SearchService()


def search_dashboard(request: HttpRequest) -> HttpResponse:
    """Renders the main semantic search interface and handles vector query execution."""
    query = request.GET.get("q", "").strip()
    results = []

    if query:
        results = search_service.semantic_search(query=query, limit=24)

    context = {
        "query": query,
        "results": results,
    }
    return render(request, "indexing/dashboard.html", context)