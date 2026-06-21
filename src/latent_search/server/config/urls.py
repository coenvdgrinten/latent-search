from django.conf import settings
from django.contrib import admin
from django.urls import include, path
from django.views import static

app_name = "indexing"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("latent_search.server.indexing.urls", namespace="indexing")),
    path(
        "media/<path:path>",
        static.serve,
        {"document_root": settings.MEDIA_ROOT},
        name="serve-media",
    ),
]
