import os

from django.http import FileResponse, Http404
from django.conf import settings

from rest_framework import generics, status
from rest_framework.response import Response
from rest_framework.views import APIView

from llm_engine.models import LLMInteraction, Document, LLMModel
from llm_engine.qa_service import QAService
from .serializers import (
    QAAskRequestSerializer,
    LLMInteractionSerializer,
    DocumentUploadSerializer,
    DocumentUpdateSerializer,
)
from llm_engine.services import DocumentIngestionService
from llm_engine.vector_store import delete_document_vectors

class QAAskAPIView(APIView):
    """
    POST /api/qa/ask/
    Body: { "question": "...", "document_ids": [1,2], "k": 5 }
    """
    def post(self, request):
        req_ser = QAAskRequestSerializer(data=request.data)
        req_ser.is_valid(raise_exception=True)
        data = req_ser.validated_data

        question = data["question"]
        document_ids = data.get("document_ids", None)
        k = data.get("k", 5)
        interaction_id = data.get("interaction_id")
        model_id = data.get("model_id")

        interaction = None
        if interaction_id:
            try:
                interaction = LLMInteraction.objects.get(id=interaction_id)
            except LLMInteraction.DoesNotExist:
                return Response(
                    {"detail": "interaction_id not found"},
                    status=status.HTTP_404_NOT_FOUND
                )

        if interaction is None:
            if model_id:
                llm_model = LLMModel.objects.get(id=model_id)
            else:
                llm_model = LLMModel.objects.filter(is_active=True).first()
                if not llm_model:
                    llm_model = LLMModel.objects.create(name="openrouter/owl-alpha", is_active=True)

            interaction = LLMInteraction.objects.create(
                prompt=question,
                status="pending",
                llm_model=llm_model,
            )

        if document_ids:
            interaction.target_documents.set(Document.objects.filter(id__in=document_ids))
        else:
            interaction.target_documents.clear()

        answer = QAService.ask(
            question=question,
            document_ids=document_ids,
            k=k,
            interaction=interaction
        )

        out = {
            "interaction": LLMInteractionSerializer(interaction).data,
            "answer": answer
        }
        return Response(out, status=status.HTTP_200_OK)


class LLMInteractionListAPIView(generics.ListAPIView):
    """
    GET /api/qa/interactions/
    """
    serializer_class = LLMInteractionSerializer

    def get_queryset(self):
        return (
            LLMInteraction.objects
            .all()
            .order_by("-created_at")
            .prefetch_related("target_documents", "context_chunks", "context_chunks__document")
        )


class LLMInteractionDetailAPIView(generics.RetrieveAPIView):
    """
    GET /api/qa/interactions/{id}/
    """
    serializer_class = LLMInteractionSerializer
    lookup_field = "id"

    def get_queryset(self):
        return (
            LLMInteraction.objects
            .all()
            .prefetch_related("target_documents", "context_chunks", "context_chunks__document")
        )


class DocumentUploadAPIView(APIView):
    """
    POST /api/documents/upload/
    """
    def post(self, request):
        serializer = DocumentUploadSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        file_obj = serializer.validated_data["file"]
        title = serializer.validated_data.get("title", file_obj.name)

        try:
            # If your model has a FileField, save it here
            doc = Document.objects.create(title=title, file=file_obj)

            # If needed, attach the file:
            # doc.file.save(file_obj.name, file_obj, save=True)

            # Process the uploaded document
            success = DocumentIngestionService.process_document(doc)

            if success:
                return Response(
                    {
                        "message": "Document uploaded and processed successfully",
                        "id": doc.id,
                    },
                    status=status.HTTP_201_CREATED,
                )

            return Response(
                {
                    "message": "Document uploaded but processing failed",
                    "id": doc.id,
                },
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

        except Exception as e:
            return Response(
                {"message": f"Error during processing: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )
        
class DocumentUpdateAPIView(APIView):
    """
    PUT /api/documents/{id}/update/
    PATCH /api/documents/{id}/update/

    If a new file is uploaded:
    - replace the file
    - delete old chunks
    - reset processed state
    - reprocess document
    """

    def put(self, request, id):
        return self._update(request, id)

    def patch(self, request, id):
        return self._update(request, id)

    def _update(self, request, id):
        try:
            doc = Document.objects.get(id=id)
        except Document.DoesNotExist:
            return Response(
                {"detail": "Document not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = DocumentUpdateSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        data = serializer.validated_data

        new_title = data.get("title")
        new_file = data.get("file")

        try:
            reprocessed = False

            if new_title is not None:
                doc.title = new_title

            if new_file is not None:
                doc.file = new_file
                doc.save()
            
                # Delegate the heavy lifting to the service
                success = DocumentIngestionService.reprocess_document(doc)
                reprocessed = True

                if not success:
                    return Response(
                        {
                            "message": "Document updated, but reprocessing failed.",
                            "id": doc.id,
                        },
                        status=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    )
            else:
                doc.save()

            return Response(
                {
                    "message": "Document updated successfully.",
                    "id": doc.id,
                    "reprocessed": reprocessed,
                },
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"message": f"Error during update: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

class DocumentDeleteAPIView(APIView):
    """
    DELETE /api/documents/{id}/delete/
    """

    def delete(self, request, id):
        try:
            doc = Document.objects.get(id=id)
        except Document.DoesNotExist:
            return Response(
                {"detail": "Document not found."},
                status=status.HTTP_404_NOT_FOUND
            )

        try:
            try:
                delete_document_vectors(id)
            except Exception as e:
                # Optional: Decide if you want to abort the deletion if vectors fail
                print(f"Warning: Failed to cleanup vectors: {e}")

            doc.delete()

            return Response(
                {"message": "Document deleted successfully."},
                status=status.HTTP_200_OK,
            )

        except Exception as e:
            return Response(
                {"message": f"Error during delete: {str(e)}"},
                status=status.HTTP_500_INTERNAL_SERVER_ERROR,
            )

def download_by_filename(request, filename):
        file_path = os.path.join(settings.MEDIA_ROOT, "documents", filename)

        if not os.path.exists(file_path):
            raise Http404("File not found")

        return FileResponse(
            open(file_path, "rb"),
            as_attachment=True,
            filename=filename
        )