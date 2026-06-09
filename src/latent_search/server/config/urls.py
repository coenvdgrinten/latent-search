from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

app_name = "indexing"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("latent_search.server.indexing.urls", namespace="indexing")),
]

if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
