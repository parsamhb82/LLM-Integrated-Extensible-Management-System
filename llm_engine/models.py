from django.db import models

class LLMModel(models.Model):
    name = models.CharField(
        max_length=255,
        default="openrouter/owl-alpha",
        help_text="The exact identifier of the model used from OpenRouter"
    )
    is_active = models.BooleanField(default=True)

    def __str__(self):
        return self.name
    

class Document(models.Model):
    title = models.CharField(max_length=255)
    file = models.FileField(upload_to='documents/') 
    content = models.TextField(blank=True, null=True) 

    is_processed = models.BooleanField(
        default=False,
        help_text="Whether the document has been chunked and indexed for retrieval."
    )
    
    uploaded_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return self.title
    
class DocumentChunk(models.Model):
    """
    For Data Science/RAG: We split documents into smaller pieces.
    This helps in "Finding content related to query".
    """
    document = models.ForeignKey(Document, on_delete=models.CASCADE, related_name="chunks")
    text = models.TextField()

    def __str__(self):
        return f"Chunk of {self.document.title}"


class LLMInteraction(models.Model):
    llm_model = models.ForeignKey(LLMModel, on_delete=models.PROTECT, related_name="interactions")

    context_chunks = models.ManyToManyField(DocumentChunk, blank=True)

    prompt = models.TextField()
    response_text = models.TextField(blank=True, null=True)

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('success', 'Success'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(
        max_length=10, 
        choices=STATUS_CHOICES, 
        default='pending'
    )

    raw_request = models.JSONField(
        blank=True, 
        null=True, 
        help_text="The complete JSON payload sent to OpenRouter."
    )
    raw_response = models.JSONField(
        blank=True, 
        null=True, 
        help_text="The complete raw JSON response returned by OpenRouter."
    )

    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Query to {self.llm_model.name} on {self.created_at}"