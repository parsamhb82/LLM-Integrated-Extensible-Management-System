from __future__ import annotations

from django.core.management.base import BaseCommand, CommandError

from llm_engine.models import Document, LLMModel
from llm_engine.qa_service import QAService


class Command(BaseCommand):
    help = "Ask a question over selected documents using RAG + OpenRouter."

    def add_arguments(self, parser):
        parser.add_argument("--question", required=True, type=str)
        parser.add_argument("--doc-ids", nargs="*", type=int, default=[])
        parser.add_argument("--k", type=int, default=5)
        parser.add_argument("--model-id", type=int, default=None)

    def handle(self, *args, **options):
        question = options["question"]
        doc_ids = options["doc_ids"]
        k = options["k"]
        model_id = options["model_id"]

        llm_model = None
        if model_id is not None:
            llm_model = LLMModel.objects.filter(id=model_id).first()
            if llm_model is None:
                raise CommandError(f"LLMModel id={model_id} not found")

        if doc_ids:
            found = set(Document.objects.filter(id__in=doc_ids).values_list("id", flat=True))
            missing = [d for d in doc_ids if d not in found]
            if missing:
                raise CommandError(f"Some doc ids do not exist: {missing}")

        result = QAService.ask(
            question=question,
            document_ids=doc_ids or None,
            k=k,
            llm_model=llm_model,
        )

        self.stdout.write(self.style.SUCCESS("Answer:"))
        self.stdout.write(result.answer_text)
        self.stdout.write("")
        self.stdout.write(f"Interaction ID: {result.interaction.id}")
        self.stdout.write(f"Chunks used: {result.chunk_ids}")
