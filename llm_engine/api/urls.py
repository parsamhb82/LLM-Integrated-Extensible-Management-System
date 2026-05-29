from django.urls import path
from .views import QAAskAPIView, LLMInteractionListAPIView, LLMInteractionDetailAPIView,DocumentUploadAPIView, DocumentUpdateAPIView, DocumentDeleteAPIView

urlpatterns = [
    path("qa/ask/", QAAskAPIView.as_view(), name="qa-ask"),
    path("qa/interactions/", LLMInteractionListAPIView.as_view(), name="qa-interaction-list"),
    path("qa/interactions/<int:id>/", LLMInteractionDetailAPIView.as_view(), name="qa-interaction-detail"),
    path("documents/upload/", DocumentUploadAPIView.as_view(), name="document-upload"),
    path("documents/<int:id>/update/", DocumentUpdateAPIView.as_view(), name="document-update"),
    path("documents/<int:id>/delete/", DocumentDeleteAPIView.as_view(), name="document-delete"),
]