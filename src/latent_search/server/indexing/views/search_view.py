from django.http import HttpRequest, HttpResponse
from django.shortcuts import render

from latent_search.server.indexing.services.search import (
    QdrantUnavailableError,
    SearchService,
)

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
            error = (
                "Search is unavailable: could not connect to the vector database."
                " Is Qdrant running?"
            )

    # HTMX requests get just the results fragment
    if request.headers.get("HX-Request"):
        template = (
            "indexing/_results_fragment.html"
            if not error
            else "indexing/_error_fragment.html"
        )
    else:
        template = "indexing/dashboard.html"

    context = {
        "query": query,
        "results": results,
        "error": error,
    }
    return render(request, template, context)
