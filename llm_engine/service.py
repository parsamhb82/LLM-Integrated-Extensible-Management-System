import os
from django.db import transaction
from docx import Document as DocxDocument
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .models import Document, DocumentChunk

class DocumentIngestionService:
    """
    Handles extracting text from PDF/DOCX files, saving the full text,
    and splitting the content into semantic chunks for RAG.
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
        """
        Extracts content from the uploaded file, saves it,
        splits it into chunks using LangChain, and updates the database.
        """
        if not document.file:
            raise ValueError(f"Document {document.id} has no uploaded file.")

        file_path = document.file.path
        _, extension = os.path.splitext(file_path.lower())

        # 1. Extract Text based on file extension
        try:
            if extension == '.pdf':
                extracted_text = cls.extract_text_from_pdf(file_path)
            elif extension == '.docx':
                extracted_text = cls.extract_text_from_docx(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        except Exception as e:
            # In production, log the error (e.g., logger.error)
            print(f"Error reading file {file_path}: {e}")
            return False

        if not extracted_text:
            print(f"No text could be extracted from {file_path}")
            return False

        # Use Django transaction to make sure chunk generation is atomic
        with transaction.atomic():
            # 2. Save extracted content to the Document
            document.content = extracted_text
            document.save()

            # 3. Clear existing chunks if this is a re-run
            document.chunks.all().delete()

            # 4. Use LangChain splitter to break text into chunks
            # A chunk size of 500-1000 characters with a small overlap is standard for basic RAG
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                length_function=len,
            )
            chunks = text_splitter.split_text(extracted_text)

            # 5. Bulk create DocumentChunks in the database for speed
            chunk_objects = [
                DocumentChunk(document=document, text=chunk_text)
                for chunk_text in chunks
            ]
            DocumentChunk.objects.bulk_create(chunk_objects)

        return True
