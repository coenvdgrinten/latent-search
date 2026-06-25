from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views import static

from latent_search.server.indexing.views import login_view, logout_view

app_name = "indexing"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("login/", login_view, name="login"),
    path("logout/", logout_view, name="logout"),
    path("", include("latent_search.server.indexing.urls", namespace="indexing")),
    path(
        "media/<path:path>",
        static.serve,
        {"document_root": settings.MEDIA_ROOT},
        name="serve-media",
    ),
]
