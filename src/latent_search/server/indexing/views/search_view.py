import base64
from pathlib import Path

from django.contrib.auth.decorators import login_required
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import render

from latent_search.server.indexing.apps import _model_ready_event
from latent_search.server.indexing.services.search import (
    QdrantUnavailableError,
    SearchService,
)

search_service = SearchService()


@login_required
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


@login_required
def model_ready_check(_request: HttpRequest) -> JsonResponse:
    """Returns JSON indicating whether the embedding model is warmed up."""
    return JsonResponse({"ready": _model_ready_event.is_set()})


@login_required
def serve_image(request: HttpRequest, b64_path: str) -> FileResponse:
    """Serve an image from an arbitrary filesystem path.

    ``b64_path`` is a base64-encoded absolute path, avoiding URL-escaping issues
    with slashes or special characters (e.g. apostrophes in folder names).
    """
    try:
        decoded = base64.b64decode(b64_path).decode()
    except Exception as exc:
        raise Http404("Invalid path encoding") from exc

    filepath = Path(decoded)
    if not filepath.exists():
        raise Http404(f"Image not found: {decoded}")

    return FileResponse(filepath.open("rb"), content_type="image/jpeg")
