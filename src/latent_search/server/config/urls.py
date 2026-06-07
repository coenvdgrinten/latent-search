from django.contrib import admin
from django.urls import include, path

app_name = "indexing"

urlpatterns = [
    path("admin/", admin.site.urls),
    path("", include("latent_search.server.indexing.urls", namespace="indexing")),
]
