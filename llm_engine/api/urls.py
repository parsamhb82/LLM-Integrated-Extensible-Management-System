from django.urls import path
from .views import QAAskAPIView, LLMInteractionListAPIView, LLMInteractionDetailAPIView,DocumentUploadAPIView

urlpatterns = [
    path("qa/ask/", QAAskAPIView.as_view(), name="qa-ask"),
    path("qa/interactions/", LLMInteractionListAPIView.as_view(), name="qa-interaction-list"),
    path("qa/interactions/<int:id>/", LLMInteractionDetailAPIView.as_view(), name="qa-interaction-detail"),
    path("documents/upload/", DocumentUploadAPIView.as_view(), name="document-upload"),
]