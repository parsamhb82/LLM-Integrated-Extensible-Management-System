from typing import Iterable, List, Optional

from django.conf import settings

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument

from .models import Document

def get_embedding_function():
    """
    Loads the embedding model used for converting text into vectors.
    """
    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

def get_vectorstore():
    """
    Returns the Chroma vector store instance.
    """
    embedding_function = get_embedding_function()

    return Chroma(
        persist_directory=str(settings.CHROMA_DIR),
        embedding_function=embedding_function,
        collection_name="document_chunks",
    )


def delete_document_vectors(document_id: int):
    """
    Deletes all vectors related to one document from Chroma.
    """
    vectorstore = get_vectorstore()

    results = vectorstore.get(
        where={"document_id": document_id}
    )

    ids = results.get("ids", [])
    if ids:
        vectorstore.delete(ids=ids)


def index_document_chunks(document: Document):
    """
    Indexes all document chunks into Chroma.
    """
    vectorstore = get_vectorstore()

    chunks = document.chunks.all().order_by("id")
    docs = []

    for chunk in chunks:
        docs.append(
            LangChainDocument(
                page_content=chunk.text,
                metadata={
                    "chunk_id": chunk.id,
                    "document_id": document.id,
                    "document_title": document.title,
                }
            )
        )

    if docs:
        vectorstore.add_documents(docs)

def _normalize_doc_ids(document_ids: Optional[Iterable[int]]) -> List[int]:
    if not document_ids:
        return []
    return sorted({int(x) for x in document_ids})

def similarity_search_chunks(
    query: str,
    k: int = 5,
    document_ids: Optional[Iterable[int]] = None,
) -> List[LangChainDocument]:
    """
    Returns top-k similar chunks for the query.

    If document_ids is provided, tries to restrict retrieval to those documents.
    Falls back to client-side filtering if the Chroma metadata filter doesn't support $in.
    """
    vectorstore = get_vectorstore()
    doc_ids = _normalize_doc_ids(document_ids)

    # No filtering requested
    if not doc_ids:
        return vectorstore.similarity_search(query=query, k=k)

    # 1) Try server-side filtering via Chroma "where" (if supported)
    # Some installations support {"document_id": {"$in": [..]}}
    try:
        return vectorstore.similarity_search(
            query=query,
            k=k,
            filter={"document_id": {"$in": doc_ids}},
        )
    except Exception:
        # 2) Fallback: over-fetch, then filter in Python
        # Fetch more than k to compensate for filtering losses.
        overfetch = max(k * 5, 25)
        docs = vectorstore.similarity_search(query=query, k=overfetch)

        filtered = [
            d for d in docs
            if int(d.metadata.get("document_id", -1)) in doc_ids
        ]
        return filtered[:k]