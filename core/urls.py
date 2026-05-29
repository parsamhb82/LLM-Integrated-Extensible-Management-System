from django.contrib import admin
from django.urls import path, include
from llm_engine.api.views import download_by_filename

urlpatterns = [
    path('admin/', admin.site.urls),
    path("api/", include("llm_engine.api.urls")),
    path("documents/<str:filename>", download_by_filename)
]
