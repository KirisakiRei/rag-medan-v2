# ============================================================
# ✅ RAG Document API — Standalone dengan Post-Summarization Toggle
# ============================================================
import logging, sys, os
from fastapi import FastAPI
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import CONFIG

# Import routers
from routes.doc_sync_routes import doc_sync_router
from routes.doc_search_routes import doc_search_router

# ============================================================
# 🔹 Setup Logging
# ============================================================
LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")
os.makedirs(LOG_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOG_DIR, "rag-doc.log")

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8", mode="a")
    ]
)
logger = logging.getLogger("doc_app")
logger.info("=== ✅ RAG Document API initialized successfully ===")

# ============================================================
# 🔹 FastAPI App
# ============================================================
app = FastAPI(title="RAG Document Service")

# ============================================================
# 🔹 Model & Qdrant setup (Global Variables)
# ============================================================
qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)
model_doc = SentenceTransformer(CONFIG["embeddings"]["model_path_large"])

# ============================================================
# 🔹 Helper: Embed Query
# ============================================================
def embed_query(model, text: str):
    """Embed query text untuk pencarian dokumen (standar e5 format)."""
    return model.encode(f"query: {text}", normalize_embeddings=True).tolist()

# ============================================================
# 🔹 Register Routers
# ============================================================
app.include_router(doc_sync_router)
app.include_router(doc_search_router)

logger.info("✅ All routers registered successfully")
logger.info("   - /api/doc-sync → doc_sync_routes.py")
logger.info("   - /api/doc-search → doc_search_routes.py")

# ============================================================
# 🔹 Run Server
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=CONFIG["doc_api"]["host"],
        port=CONFIG["doc_api"]["port"],
        log_config=None
    )
