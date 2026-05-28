import os

from django.db import transaction

from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Document, DocumentChunk
from .vector_store import index_document_chunks, delete_document_vectors

class DocumentIngestionService:
    """
    Handles extracting text from PDF/DOCX files, saving the full text,
    splitting the content into semantic chunks for RAG,
    and indexing them in the vector store.
    """

    @staticmethod
    def extract_text_from_pdf(file_path) -> str:
        """Extracts all text from a PDF file."""
        text = ""
        reader = PdfReader(file_path)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text.strip()

    @staticmethod
    def extract_text_from_docx(file_path) -> str:
        """Extracts all text from a DOCX file."""
        doc = DocxDocument(file_path)
        text = []
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text.append(paragraph.text)
        return "\n".join(text).strip()

    @classmethod
    def process_document(cls, document: Document) -> bool:
        if not document.file:
            raise ValueError(f"Document {document.id} has no uploaded file.")

        file_path = document.file.path
        _, extension = os.path.splitext(file_path.lower())

        try:
            if extension == ".pdf":
                extracted_text = cls.extract_text_from_pdf(file_path)
            elif extension == ".docx":
                extracted_text = cls.extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
            document.is_processed = False
            document.save(update_fields=["is_processed", "updated_at"])
            return False

        if not extracted_text:
            print(f"No text could be extracted from {file_path}")
            document.is_processed = False
            document.save(update_fields=["is_processed", "updated_at"])
            return False

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=100,
            length_function=len,
        )
        chunks = text_splitter.split_text(extracted_text)

        if not chunks:
            print(f"No chunks were generated for document {document.id}")
            document.is_processed = False
            document.save(update_fields=["is_processed", "updated_at"])
            return False

        try:
            # mark as not processed before starting replacement
            document.is_processed = False
            document.save(update_fields=["is_processed", "updated_at"])

            # delete old vectors first
            delete_document_vectors(document.id)

            with transaction.atomic():
                document.content = extracted_text
                document.save(update_fields=["content", "updated_at"])

                document.chunks.all().delete()

                chunk_objects = [
                    DocumentChunk(document=document, text=chunk_text)
                    for chunk_text in chunks
                ]
                DocumentChunk.objects.bulk_create(chunk_objects)

            # only after DB transaction commits successfully
            index_document_chunks(document)

            document.is_processed = True
            document.save(update_fields=["is_processed", "updated_at"])
            return True

        except Exception as e:
            print(f"Error processing document {document.id}: {e}")
            document.is_processed = False
            document.save(update_fields=["is_processed", "updated_at"])
            return False