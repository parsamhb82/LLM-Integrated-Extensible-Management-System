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
        model_name=settings.EMBEDDING_MODEL_NAME
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