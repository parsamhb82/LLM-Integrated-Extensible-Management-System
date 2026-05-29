from django.conf import settings

from .models import LLMInteraction, LLMModel, DocumentChunk
from .retrieval import RetrievalService
from .llm_client import OpenRouterClient

class QAService:
    @staticmethod
    def ask(question, document_ids=None, k=5, interaction=None):
        # 1. Get the Model configuration from DB
        # Make sure you have a record in LLMModel with name "openrouter/owl-alpha"
        llm_config = LLMModel.objects.filter(is_active=True).first()
        model_name = llm_config.name if llm_config else "openrouter/owl-alpha"

        # 2. Retrieve relevant chunks from Chroma
        retrieved_docs = RetrievalService.retrieve_context(
            question=question, 
            document_ids=document_ids, 
            k=k
        )

        # 3. Format context string
        context_parts = []
        chunk_ids = []
        for doc in retrieved_docs:
            cid = doc.metadata.get("chunk_id")
            if cid: chunk_ids.append(cid)
            context_parts.append(f"document content:\n{doc.page_content}")
        
        context_text = "\n\n---\n\n".join(context_parts)

        # 4. Prepare Messages (System + User)
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful document question‑answering assistant.\n"
                    "Answer strictly and only based on the provided context.\n"
                    "If the answer cannot be found in the context, say that you do not know.\n"
                    "Answer in the same language as the user's question.\n"
                    "Do not add information that is not explicitly supported by the context."
                )
            },
            {
                "role": "user", 
                "content": f"Context:\n{context_text}\n\nQuestion: {question}"
            }
        ]

        # 5. Use existing interaction or create a new one
        if interaction is None:
            interaction = LLMInteraction.objects.create(
                llm_model=llm_config,
                prompt=question,
                status='pending'
            )
        else:
            # Update the existing interaction's status and model
            if not interaction.llm_model and llm_config:
                interaction.llm_model = llm_config
            interaction.status = 'pending'
            interaction.save()

        # Link retrieved chunks to the interaction
        if chunk_ids:
            interaction.context_chunks.set(DocumentChunk.objects.filter(id__in=chunk_ids))
        else:
            interaction.context_chunks.clear()

        # 6. Call OpenRouter
        client = OpenRouterClient(api_key=settings.OPENROUTER_API_KEY, model=model_name)
        
        try:
            raw_response = client.chat_completions(messages)
            answer = raw_response['choices'][0]['message']['content']
            
            # Update interaction with success
            interaction.response_text = answer
            interaction.raw_request = {"messages": messages, "model": model_name}
            interaction.raw_response = raw_response
            interaction.status = 'success'
            interaction.save()
            
            return answer

        except Exception as e:
            interaction.status = 'failed'
            interaction.response_text = str(e)
            interaction.save()
            raise e
