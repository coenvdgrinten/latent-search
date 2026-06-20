from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

app_name = "indexing"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("latent_search.server.indexing.urls", namespace="indexing")),
]

# Serve static + media files (needed behind Gunicorn where DEBUG=False)
urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
