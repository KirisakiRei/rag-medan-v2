from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import logging, sys, time
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

try:
    from config import CONFIG
except Exception:
    CONFIG = {
        "doc_api": {"host": "0.0.0.0", "port": 5100},
        "qdrant": {"host": "localhost", "port": 6333},
        "embeddings": {"model_path_large": "/home/kominfo/models/multilingual-e5-large"},
        "ocr": {"engine": "paddle", "lang": "id"}
    }

from core.document_pipeline import process_document
from core.embedding_utils import embed_query

LOG_FILE = "./logs/rag-doc.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("doc_app")

app = FastAPI(title="RAG Document Service")

qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])
model_doc = SentenceTransformer(CONFIG["embeddings"]["model_path_large"])

class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: Optional[str] = None
    file_url: str


class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5

@app.get("/health")
async def health():
    try:
        _ = model_doc.encode("health", normalize_embeddings=True)
        _ = qdrant.get_collections()
        return {"status": "healthy", "components": {"model_doc": True, "qdrant": True}}
    except Exception as e:
        logger.exception("Health check error")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/doc-sync")
async def doc_sync(req: DocSyncRequest):
    try:
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
            raise HTTPException(status_code=500, detail=result)
        return result
    except Exception as e:
        logger.exception("doc-sync error")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/doc-search")
async def doc_search(req: DocSearchRequest):
    try:
        vec = embed_query(model_doc, req.query)
        hits = qdrant.search(collection_name="document_bank", query_vector=vec, limit=req.limit)
        results = [{
            "doc_id": h.payload.get("doc_id"),
            "category": h.payload.get("category"),
            "chunk_index": h.payload.get("chunk_index"),
            "text": h.payload.get("text"),
            "score": float(h.score)
        } for h in hits]
        return {"status": "success", "results": results}
    except Exception as e:
        logger.exception("doc-search error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["doc_api"]["host"], port=CONFIG["doc_api"]["port"])