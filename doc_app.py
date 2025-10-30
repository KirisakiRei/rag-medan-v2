# doc_app.py (potongan penting)

import logging, sys
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import CONFIG

from core.document_pipeline import process_document
from core.embedding_utils import embed_query  # bila dipakai

LOG_FILE = "./logs/rag-doc.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(LOG_FILE, encoding="utf-8")]
)
logger = logging.getLogger("doc_app")

app = FastAPI(title="RAG Document Service")

# config ...
qdrant = QdrantClient(host=CONFIG["qdrant"]["host"], port=CONFIG["qdrant"]["port"])
model_doc = SentenceTransformer(CONFIG["embeddings"]["model_path_large"])

class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: str | None = None
    file_url: str

@app.post("/api/doc-sync")
async def doc_sync(req: DocSyncRequest):
    """
    Blocking endpoint: proses akan berjalan sinkron dan response dikirim setelah selesai.
    Logging progres akan ditulis ke LOG_FILE dan stdout sehingga kamu bisa `tail -f`.
    """
    try:
        logger.info(f"[API] doc-sync request doc_id={req.doc_id} opd={req.opd_name} url={req.file_url}")
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
            logger.warning(f"[API] doc-sync result not ok: {result}")
            raise HTTPException(status_code=500, detail=result)
        logger.info(f"[API] doc-sync finished doc_id={req.doc_id} chunks={result.get('total_chunks')}")
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[API] doc-sync error")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host=CONFIG["doc_api"]["host"], port=CONFIG["doc_api"]["port"])