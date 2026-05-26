from django.contrib import admin

from .models import LLMModel, Document, DocumentChunk, LLMInteraction
from .services import DocumentIngestionService

@admin.register(LLMModel)
class LLMModelAdmin(admin.ModelAdmin):
    list_display = ("id", "name", "is_active")
    list_filter = ("is_active",)
    search_fields = ("name",)
    ordering = ("name",)

class DocumentChunkInline(admin.TabularInline):
    model = DocumentChunk
    extra = 0
    readonly_fields = ("text",)
    can_delete = False
    show_change_link = True


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ("id", "title", "uploaded_at", "updated_at")
    search_fields = ("title", "content")
    readonly_fields = ("uploaded_at", "updated_at")
    inlines = [DocumentChunkInline]

    actions = ["process_documents"]

    def process_documents(self, request, queryset):
        success_count = 0
        failed_count = 0

        for doc in queryset:
            success = DocumentIngestionService.process_document(doc)
            if success:
                success_count += 1
            else:
                failed_count += 1

        if success_count > 0:
            self.message_user(
                request, 
                f"Successfully parsed and chunked {success_count} document(s) and failed {failed_count} document(s)"
            )
        if failed_count > 0:
            self.message_user(
                request, 
                f"Failed to process {failed_count} document(s). See logs for details.",
                level='ERROR'
            )

    process_documents.short_description = "Process / Chunk selected documents"
    
@admin.register(DocumentChunk)
class DocumentChunkAdmin(admin.ModelAdmin):
    list_display = ("id", "document", "short_text")
    search_fields = ("text", "document__title")
    list_filter = ("document",)

    def short_text(self, obj):
        return obj.text[:80] + "..." if len(obj.text) > 80 else obj.text
    short_text.short_description = "Text preview"


@admin.register(LLMInteraction)
class LLMInteractionAdmin(admin.ModelAdmin):
    list_display = ("id", "llm_model", "status", "created_at")
    list_filter = ("status", "llm_model", "created_at")
    search_fields = ("prompt", "response_text", "llm_model__name")
    readonly_fields = ("created_at",)
    filter_horizontal = ("context_chunks",)