from django.urls import path

from . import views

app_name = "indexing"

urlpatterns = [
    path("", views.search_dashboard, name="dashboard"),
    path("ready", views.model_ready_check, name="ready"),
    path("image/<str:b64_path>/", views.serve_image, name="serve-image"),
    # Admin / operations APIs
    path("api/stats", views.get_stats, name="api-stats"),
    path("api/start_job", views.start_job, name="api-start-job"),
    path("api/stop_job", views.stop_job, name="api-stop-job"),
    path("api/save_settings", views.save_settings, name="api-save-settings"),
    # Library transfer (cross-machine workflow)
    path("api/export_library", views.export_library, name="api-export-library"),
    path("api/import_library", views.import_library, name="api-import-library"),
]
