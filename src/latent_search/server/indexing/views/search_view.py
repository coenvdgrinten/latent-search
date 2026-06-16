from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from latent_search.server.indexing.apps import _model_ready_event
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
        "model_ready": _model_ready_event.is_set(),
    }
    return render(request, template, context)


def model_ready_check(_request: HttpRequest) -> JsonResponse:
    """Returns JSON indicating whether the embedding model is warmed up."""
    return JsonResponse({"ready": _model_ready_event.is_set()})
