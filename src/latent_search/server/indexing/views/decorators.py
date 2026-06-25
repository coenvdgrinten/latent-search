"""Authentication decorator for JSON API endpoints."""

import functools
from typing import Any

from django.http import HttpRequest, JsonResponse


def login_required_json(view_func):
    """Like @login_required but returns 401 JSON instead of redirecting."""
    @functools.wraps(view_func)
    def wrapper(request: HttpRequest, *args: Any, **kwargs: Any):
        if not request.user.is_authenticated:
            return JsonResponse({"error": "Authentication required"}, status=401)
        return view_func(request, *args, **kwargs)
    return wrapper