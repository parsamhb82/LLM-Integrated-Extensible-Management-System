from django.contrib import admin

from .models import LLMModel, Document, DocumentChunk, LLMInteraction
from .services import DocumentIngestionService
from .qa_service import QAService


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
    list_display = ('prompt', 'status', 'created_at')
    readonly_fields = ('status', 'response_text', 'raw_request', 'raw_response', 'context_chunks')
    filter_horizontal = ('target_documents', 'context_chunks')
    
    actions = ['run_qa_action']

    @admin.action(description="Run Q&A on selected interactions")
    def run_qa_action(self, request, queryset):
        for interaction in queryset:
            try:
                # Extract doc IDs from the M2M field
                doc_ids = list(interaction.target_documents.values_list('id', flat=True))
                
                # Use your existing QAService
                # Note: We are overwriting the interaction logic here
                answer = QAService.ask(
                    question=interaction.prompt,
                    document_ids=doc_ids if doc_ids else None,
                    k=5,
                    interaction=interaction
                )
                self.message_user(request, f"Successfully processed: {interaction.prompt[:20]}...")
            except Exception as e:
                self.message_user(request, f"Error processing {interaction.prompt[:20]}: {str(e)}", level='ERROR')