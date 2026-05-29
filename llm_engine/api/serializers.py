from rest_framework import serializers
from llm_engine.models import LLMInteraction, Document, DocumentChunk


class QAAskRequestSerializer(serializers.Serializer):
    question = serializers.CharField(max_length=4000)
    document_ids = serializers.ListField(
        child=serializers.IntegerField(min_value=1),
        required=False,
        allow_null=True
    )
    k = serializers.IntegerField(required=False, min_value=1, max_value=20, default=5)

    interaction_id = serializers.IntegerField(required=False, min_value=1)
    model_id = serializers.IntegerField(required=False)

    def validate_model_id(self, value):
        from llm_engine.models import LLMModel
        if not LLMModel.objects.filter(id=value).exists():
            raise serializers.ValidationError("LLMModel with this ID does not exist.")
        return value

    def validate_document_ids(self, value):
        if value is None:
            return None
        # de-dup
        value = list(dict.fromkeys(value))
        # validate existence
        existing = set(Document.objects.filter(id__in=value).values_list("id", flat=True))
        missing = [i for i in value if i not in existing]
        if missing:
            raise serializers.ValidationError(f"Invalid document_ids: {missing}")
        return value


class DocumentMiniSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = ("id", "title")


class DocumentUploadSerializer(serializers.Serializer):
    file = serializers.FileField()
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)

    def validate(self, attrs):
        file_obj = attrs["file"]
        title = attrs.get("title")

        if not title:  # None or "" -> fallback
            attrs["title"] = file_obj.name

        return attrs



class ChunkFullSerializer(serializers.ModelSerializer):
    document_id = serializers.IntegerField(source="document.id", read_only=True)
    document_title = serializers.CharField(source="document.title", read_only=True)

    class Meta:
        model = DocumentChunk
        fields = ("id", "document_id", "document_title", "text")


class LLMInteractionSerializer(serializers.ModelSerializer):
    target_documents = DocumentMiniSerializer(many=True, read_only=True)
    context_chunks = ChunkFullSerializer(many=True, read_only=True)

    class Meta:
        model = LLMInteraction
        fields = (
            "id",
            "prompt",
            "status",
            "response_text",
            "created_at",
            "target_documents",
            "context_chunks",
        )


class DocumentUpdateSerializer(serializers.Serializer):
    file = serializers.FileField(required=False)
    title = serializers.CharField(max_length=255, required=False, allow_blank=True)

    def validate(self, attrs):
        if not attrs:
            raise serializers.ValidationError("At least one of 'title' or 'file' must be provided.")

        if "title" in attrs and not attrs["title"]:
            if "file" in attrs:
                attrs["title"] = attrs["file"].name
            else:
                raise serializers.ValidationError({"title": "Title cannot be blank unless file is provided."})

        return attrs