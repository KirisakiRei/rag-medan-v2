# RAG Medan — Hybrid (Flask + FastAPI)

Dua service:
- **Flask** (`app.py`, port 5000): RAG utama (knowledge_bank).
- **FastAPI** (`doc_app.py`, port 5100): OCR + RAG dokumen (document_bank).

## Konfigurasi
Atur `config.py` (path model lokal & OCR):
```python
CONFIG = {
  "api": {"host": "0.0.0.0", "port": 5000},
  "doc_api": {"host": "0.0.0.0", "port": 5100},
  "qdrant": {"host": "localhost", "port": 6333},
  "embeddings": {
    "model_path": "/home/kominfo/models/multilingual-e5-small",
    "model_path_large": "/home/kominfo/models/multilingual-e5-large"
  },
  "ocr": {"engine": "paddle", "lang": "id"}
}
```

## Install
```bash
pip install -r requirements.txt
```

## Jalankan
```bash
# RAG utama (Flask)
gunicorn -w 4 -t 120 -b 0.0.0.0:5000 app:app

# RAG dokumen (FastAPI)
uvicorn doc_app:app --host 0.0.0.0 --port 5100
```

## API
- `POST /api/doc-sync` (FastAPI): sinkron dokumen dari WA Manajemen
  ```json
  { "doc_id": "uuid", "opd_name": "DISDUKCAPIL", "category": "Kependudukan", "file_url": "https://..." }
  ```

- `POST /api/doc-search` (FastAPI): cari di document_bank
  ```json
  { "query": "syarat KTP", "limit": 5 }
  ```

- `POST /api/search` (Flask): RAG utama; auto-fallback ke `document_bank` saat low_confidence.
```


## Struktur Folder Lengkap
```
rag-medan-hybrid/
├── app.py
├── doc_app.py
├── config.py
├── core/
│   ├── __init__.py
│   ├── db.py
│   ├── filtering.py
│   ├── relevance.py
│   ├── utils.py
│   ├── embedding_utils.py
│   ├── ocr_utils.py
│   └── document_pipeline.py
├── routes/
│   ├── __init__.py
│   ├── search_routes.py
│   ├── sync_routes.py
│   └── health_routes.py
├── logs/
│   ├── rag-main.log
│   ├── rag-doc.log
│   └── rag-summary.log
├── var/
│   └── log/
│       ├── rag-main.log
│       ├── rag-doc.log
│       └── rag-summary.log
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```
