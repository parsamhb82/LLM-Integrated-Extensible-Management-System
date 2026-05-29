# LLM Integrated Extensible Management System

A Django REST Framework backend for document ingestion, vector search, and LLM-powered question answering.

This project lets you:

- upload documents
- process and embed them into a Chroma vector database
- update and reprocess documents
- delete documents and clean up vectors
- ask questions over one or more selected documents using an LLM through OpenRouter

---

## Features

- **Document Upload**
  - Upload a file and store it in Django
  - Extract and process content
  - Chunk and index the document into ChromaDB

- **Document Update**
  - Update document title
  - Replace the file
  - Reprocess and reindex the document after file replacement

- **Document Delete**
  - Remove vectors from ChromaDB
  - Delete the document from the database

- **Q&A over Documents**
  - Retrieve relevant chunks from indexed documents
  - Send context to an LLM
  - Store prompt/response interaction history

- **Admin Panel Support**
  - Manage documents, interactions, and models from Django admin

---

## Tech Stack

- **Python 3**
- **Django 6**
- **Django REST Framework**
- **SQLite** (default database)
- **ChromaDB** for vector storage
- **Sentence Transformers** for embeddings
- **OpenRouter API** for LLM access

---

## Project Structure

A typical structure for this project is:

```text
project-root/
├── core/
│   ├── settings.py
│   ├── urls.py
│   └── ...
├── llm_engine/
│   ├── admin.py
│   ├── models.py
│   ├── services.py
│   ├── qa_service.py
│   ├── vector_store.py
│   ├── retrieval.py
│   ├── llm_client.py
│   └── api/
│       ├── serializers.py
│       ├── urls.py
│       └── views.py
├── chroma_db/
├── media/
├── manage.py
├── .env
└── README.md
```

---

## Setup & Run (Local Development)

### 1) Prerequisites
- Python 3.10+
- `pip` and `venv`

### 2) Clone the repository
```bash
git clone https://github.com/parsamhb82/LLM-Integrated-Extensible-Management-System.git
cd LLM-Integrated-Extensible-Management-System
```

### 3) Create and activate a virtual environment
Linux / macOS:
```bash
python3 -m venv llm_sys_env
source llm_sys_env/bin/activate
```

Windows (PowerShell):
```powershell
python -m venv llm_sys_env
.\llm_sys_env\Scripts\Activate.ps1
```

### 4) Install dependencies
```bash
pip install -r requirements.txt
```

### 5) Create `.env`
Create a `.env` file in the project root:

```env
DEBUG=True
SECRET_KEY=your-secret-key-here
OPENROUTER_API_KEY=your-openrouter-api-key
```

### 6) Run migrations
```bash
python3 manage.py makemigrations
python3 manage.py migrate
```

### 7) Create a superuser
```bash
python3 manage.py createsuperuser
```

### 8) Run the server
```bash
python3 manage.py runserver
```

Open: `http://127.0.0.1:8000/`

---

## Troubleshooting

### ChromaDB
Ensure the process has write permissions for the `chroma_db` directory in your project root.

### OpenRouter
Verify `OPENROUTER_API_KEY` is set correctly in your environment and restart the server if updated.

---

## Production Notes
- Set `DEBUG=False`
- Use a production-ready database (PostgreSQL)
- Serve media files using dedicated storage (e.g., AWS S3 or Nginx)
- Enforce authentication via Django Rest Framework permissions

## Run-docker

```bash
docker compose up -d --build
```
