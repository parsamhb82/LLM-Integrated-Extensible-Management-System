from typing import Iterable, List, Optional

from langchain_core.documents import Document as LangChainDocument

from .vector_store import similarity_search_chunks


class RetrievalService:
    @staticmethod
    def retrieve_context(
        question: str,
        document_ids: Optional[Iterable[int]] = None,
        k: int = 5,
    ) -> List[LangChainDocument]:
        """
        Retrieve top-k relevant chunks for a question, optionally restricted to document_ids.
        """
        question = (question or "").strip()
        if not question:
            return []

        return similarity_search_chunks(
            query=question,
            k=k,
            document_ids=document_ids,
        )
