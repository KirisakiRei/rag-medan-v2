from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

doc_sync_router = APIRouter()
logger = logging.getLogger("doc_app")


class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: str | None = None
    file_url: str


@doc_sync_router.post("/api/doc-sync")
async def doc_sync(req: DocSyncRequest):
    """Proses OCR + chunking + indexing dokumen ke Qdrant."""
    try:
        # Import dependencies dari doc_app context
        from doc_app import qdrant, model_doc
        from config import CONFIG
        from core.document_pipeline import process_document
        
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
