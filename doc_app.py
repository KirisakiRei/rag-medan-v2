# ============================================================
# ✅ FINAL FIXED LOGGING VERSION — tanpa ubah logic apapun
# ============================================================
import logging, sys, os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import CONFIG
from qdrant_client.models import Filter, FieldCondition, MatchValue
from core.document_pipeline import process_document
from core.embedding_utils import embed_query  # bila dipakai

# ============================================================
# 🔹 Setup Logging
# ============================================================
LOG_FILE = "./logs/rag-doc.log"

# pastikan folder logs ada
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

# hapus handler default agar tidak bentrok dengan uvicorn
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

# konfigurasi logging
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
# 🔹 Model & Qdrant setup
# ============================================================
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])
model_doc = SentenceTransformer(CONFIG["embeddings"]["model_path_large"])

# ============================================================
# 🔹 Request Models
# ============================================================
class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: str | None = None
    file_url: str


# ============================================================
# 🔹 Document Sync API
# ============================================================
@app.post("/api/doc-sync")
async def doc_sync(req: DocSyncRequest):
    try:
        logger.info(f"[API] doc-sync START → doc_id={req.doc_id}, opd={req.opd_name}, url={req.file_url}")
        result = process_document(
            doc_id=req.doc_id,
            opd=req.opd_name,
            file_url=req.file_url,
            qdrant=qdrant,
            model=model_doc,
            lang=CONFIG.get("ocr", {}).get("lang", "id"),
            collection_name="document_bank"
        )
        if result.get("status") != "ok":
            logger.warning(f"[API] doc-sync result NOT OK: {result}")
            raise HTTPException(status_code=500, detail=result)
        logger.info(f"[API] ✅ doc-sync FINISHED doc_id={req.doc_id} | chunks={result.get('total_chunks')}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[API] ❌ doc-sync error")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔹 Search API
# ============================================================
class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5


@app.post("/api/doc-search")
async def doc_search(req: DocSearchRequest):
    try:
        logger.info(f"[API] 🔍 doc-search query='{req.query}' limit={req.limit}")
        vec = embed_query(model_doc, req.query)
        hits = qdrant.query_points(
            collection_name="document_bank",
            query=vec,
            limit=req.limit
        )
        results = [{
            "doc_id": h.payload.get("mysql_id"),
            "opd": h.payload.get("opd"),
            "filename": h.payload.get("filename"),
            "page_number": h.payload.get("page_number"),
            "chunk_index": h.payload.get("chunk_index"),
            "text": h.payload.get("text"),
            "score": float(h.score)
        } for h in hits]
        logger.info(f"[API] ✅ doc-search results={len(results)} hits")
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("❌ doc-search error")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# 🔹 Run Server
# ============================================================
if __name__ == "__main__":
    uvicorn.run(
        app,
        host=CONFIG["doc_api"]["host"],
        port=CONFIG["doc_api"]["port"],
        log_config=None  # 👈 penting supaya logger custom kamu tidak di-replace uvicorn
    )
