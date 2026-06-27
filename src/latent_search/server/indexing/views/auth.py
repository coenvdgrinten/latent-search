"""Authentication views (login / logout)."""

import logging

from django.contrib.auth import logout
from django.http import HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import redirect, render
from django.views.decorators.csrf import csrf_exempt

logger = logging.getLogger(__name__)


@csrf_exempt
def login_view(request: HttpRequest) -> HttpResponse:
    """Render the login page and handle POST authentication.
    CSRF is disabled here so programmatic clients (offload_index) can log in
    without first scraping the CSRF token from the login form."""
    if request.user.is_authenticated:
        return redirect("/")

    if request.method == "POST":
        username = request.POST.get("username", "").strip()
        password = request.POST.get("password", "").strip()
        from django.contrib.auth import authenticate

        user = authenticate(request, username=username, password=password)
        if user is not None:
            from django.contrib.auth import login

            login(request, user)
            return redirect("/")
        # Invalid login — show error
        return render(
            request,
            "indexing/login.html",
            {"error": "Invalid username or password"},
        )

    return render(request, "indexing/login.html")


def logout_view(request: HttpRequest) -> HttpResponse | JsonResponse:
    """Log the user out. Returns JSON for API calls, redirect otherwise."""
    logout(request)
    if request.headers.get("HX-Request"):
        return JsonResponse({"redirect": "/login/"})
    return redirect("/login/")
