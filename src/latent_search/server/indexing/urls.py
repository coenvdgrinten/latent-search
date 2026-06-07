from django.urls import path

from . import views

app_name = "indexing"

urlpatterns = [
    path("", views.search_dashboard, name="dashboard"),
]