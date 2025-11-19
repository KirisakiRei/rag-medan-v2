# Routes Directory

Direktori ini berisi semua endpoint handlers yang terorganisir berdasarkan fungsinya.

## 📂 Struktur Routes

### Flask Blueprints (untuk `app.py`)

#### 1. `health_routes.py`
- **Endpoint:** `GET /health`
- **Fungsi:** Health check monitoring sistem
- **Response:** Status Flask, embedding model, dan Qdrant

#### 2. `search_routes.py`
- **Endpoint:** `POST /api/search`
- **Fungsi:** Pencarian di RAG text dengan fallback ke RAG dokumen
- **Features:**
  - AI pre-filter untuk validasi pertanyaan
  - Vector search di knowledge_bank
  - AI relevance check
  - Scoring system (dense + overlap)
  - Automatic fallback ke RAG dokumen jika score < 0.85
  - Consistent payload format

#### 3. `sync_routes.py`
- **Endpoint:** `POST /api/sync`
- **Fungsi:** Sinkronisasi data knowledge_bank
- **Actions:**
  - `bulk_sync` - Sinkronisasi massal
  - `add` - Tambah data baru
  - `update` - Update data existing
  - `delete` - Hapus data

#### 4. `usulan_routes.py`
- **Endpoints:**
  - `POST /api/sync-usulan` - Sinkronisasi data usulan_bank
  - `POST /api/search-usulan` - Pencarian usulan
- **Features:**
  - AI pre-filter (reformulasi input)
  - AI topic relevance check
  - Scoring system
  - Detailed logging

### FastAPI Routers (untuk `doc_app.py`)

#### 5. `doc_sync_routes.py`
- **Endpoint:** `POST /api/doc-sync`
- **Fungsi:** OCR + chunking + indexing dokumen ke Qdrant
- **Process:**
  - OCR multi-page
  - Chunking dengan RecursiveCharacterTextSplitter
  - Smart merge chunk pendek
  - Embedding
  - Upload ke document_bank

#### 6. `doc_search_routes.py`
- **Endpoint:** `POST /api/doc-search`
- **Fungsi:** Pencarian berbasis dokumen (hasil OCR)
- **Features:**
  - Vector search di document_bank
  - Optional post-summarization (toggle via env)
  - Direct mode (tanpa summary)
  - Post-summary mode (LLM ringkas top-k results)

## 🔄 How It Works

### Flask App (`app.py`)
```python
# app.py mengimport dan register semua blueprints
from routes.health_routes import health_bp
from routes.search_routes import search_bp
from routes.sync_routes import sync_bp
from routes.usulan_routes import usulan_bp

app.register_blueprint(health_bp)
app.register_blueprint(search_bp)
app.register_blueprint(sync_bp)
app.register_blueprint(usulan_bp)
```

### FastAPI App (`doc_app.py`)
```python
# doc_app.py mengimport dan include semua routers
from routes.doc_sync_routes import doc_sync_router
from routes.doc_search_routes import doc_search_router

app.include_router(doc_sync_router)
app.include_router(doc_search_router)
```

## ✅ Advantages

1. **Separation of Concerns** - Setiap endpoint punya file sendiri
2. **Maintainability** - Mudah mencari dan edit endpoint spesifik
3. **Scalability** - Mudah menambah endpoint baru
4. **Clean Code** - app.py dan doc_app.py jadi lebih ringkas
5. **Reusability** - Routes bisa di-reuse di aplikasi lain
6. **Testing** - Mudah untuk unit testing per endpoint

## 📝 Naming Convention

- **Flask:** `<feature>_routes.py` → exports `<feature>_bp` (Blueprint)
- **FastAPI:** `<feature>_routes.py` → exports `<feature>_router` (APIRouter)

## 🔍 Debugging

Jika ada masalah dengan endpoint:
1. Check log file untuk error message
2. Verify blueprint/router registration di app.py/doc_app.py
3. Check import statements di routes file
4. Ensure global variables (model, qdrant) accessible via `from app import model, qdrant`

## 📊 Logging

Semua routes menggunakan logger yang sama:
- **Flask routes:** `logger = logging.getLogger("app")`
- **FastAPI routes:** `logger = logging.getLogger("doc_app")`
- Log files:
  - `/var/log/rag-medan.log` - Main app log
  - `/var/log/rag-summary.log` - Summary log
  - `./logs/rag-doc.log` - Document API log
