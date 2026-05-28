# llm_core/qa_service.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

from django.conf import settings
from django.db import transaction
from django.utils import timezone

from llm_engine.llm_client import OpenRouterClient
from llm_engine.models import DocumentChunk, LLMInteraction, LLMModel
from llm_engine.retrieval import RetrievalService


@dataclass
class QAResult:
    interaction: LLMInteraction
    answer_text: str
    chunk_ids: List[int]


class QAService:
    @staticmethod
    def _build_context(chunks: List[Tuple[int, str, dict]]) -> str:
        """
        chunks: list of (chunk_id, text, metadata)
        """
        parts = []
        for i, (chunk_id, text, meta) in enumerate(chunks, start=1):
            title = meta.get("document_title", "")
            doc_id = meta.get("document_id", "")
            parts.append(
                f"[{i}] (document_id={doc_id}, title={title}, chunk_id={chunk_id})\n{text}".strip()
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def ask(
        *,
        question: str,
        document_ids: Optional[Iterable[int]] = None,
        k: int = 5,
        llm_model: Optional[LLMModel] = None,
        temperature: float = 0.2,
        max_tokens: int = 1000,
    ) -> QAResult:
        question = (question or "").strip()
        if not question:
            raise ValueError("question is required")

        # Pick model (admin can select later; for now pick active/default)
        if llm_model is None:
            llm_model = LLMModel.objects.filter(is_active=True).order_by("id").first()
        if llm_model is None:
            raise RuntimeError("No active LLMModel found. Create one in Django Admin.")

        # 1) Retrieve context chunks (LangChain docs)
        docs = RetrievalService.retrieve_context(
            question=question,
            document_ids=document_ids,
            k=k,
        )

        # Extract chunk_ids + texts
        chunk_ids: List[int] = []
        chunk_payload: List[Tuple[int, str, dict]] = []
        for d in docs:
            meta = getattr(d, "metadata", {}) or {}
            text = getattr(d, "page_content", "") or ""
            cid = int(meta.get("chunk_id", -1))
            if cid > 0:
                chunk_ids.append(cid)
            chunk_payload.append((cid, text, meta))

        context_text = QAService._build_context(chunk_payload)

        # 2) Build messages
        system_msg = (
            "You are a helpful assistant for document Q&A.\n"
            "Answer ONLY based on the provided context. If the answer is not in the context, say you don't know.\n"

        )
        user_msg = (
            f"Context:\n{context_text}\n\n"
            f"Question:\n{question}\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]

        # 3) Create interaction as pending, then call LLM, then update
        interaction = LLMInteraction.objects.create(
            llm_model=llm_model,
            prompt=user_msg,
            status="pending",
            created_at=timezone.now(),  # safe even though auto_now_add exists
        )

        # attach context chunks (if they exist in DB)
        if chunk_ids:
            db_chunks = list(DocumentChunk.objects.filter(id__in=chunk_ids))
            interaction.context_chunks.add(*db_chunks)

        client = OpenRouterClient(
            api_key=getattr(settings, "OPENROUTER_API_KEY", ""),
            model=llm_model.name,
            site_url=getattr(settings, "OPENROUTER_SITE_URL", None),
            app_name=getattr(settings, "OPENROUTER_APP_NAME", None),
        )
        if not client.api_key:
            interaction.status = "failed"
            interaction.response_text = "Missing OPENROUTER_API_KEY in settings/environment."
            interaction.save(update_fields=["status", "response_text"])
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        raw_request = {
            "model": llm_model.name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            raw_response = client.chat_completions(
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            answer_text = (
                raw_response.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )

            # Save results
            interaction.status = "success"
            interaction.response_text = answer_text
            interaction.raw_request = raw_request
            interaction.raw_response = raw_response
            interaction.save(update_fields=["status", "response_text", "raw_request", "raw_response"])

        except Exception as e:
            interaction.status = "failed"
            interaction.response_text = f"LLM call failed: {e}"
            interaction.raw_request = raw_request
            interaction.save(update_fields=["status", "response_text", "raw_request"])
            raise

        return QAResult(interaction=interaction, answer_text=answer_text, chunk_ids=chunk_ids)
