from django.contrib import admin
from .models import LLMModel, Document, DocumentChunk, LLMInteraction

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

    actions = ["mark_for_reprocessing"]

    def mark_for_reprocessing(self, request, queryset):
        # Placeholder action for future ingestion pipeline
        count = queryset.count()
        self.message_user(
            request,
            f"{count} document(s) selected for reprocessing. Implement ingestion service next."
        )
    mark_for_reprocessing.short_description = "Mark selected documents for reprocessing"
    
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