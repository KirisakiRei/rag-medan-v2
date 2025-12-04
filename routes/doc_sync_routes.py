from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging
from doc_app import qdrant, model_doc
from config import CONFIG
from core.document_pipeline import process_document

doc_sync_router = APIRouter()
logger = logging.getLogger("doc_app")


class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: str | None = None
    file_url: str


@doc_sync_router.post("/api/doc-sync")
async def doc_sync(sync_request: DocSyncRequest):
    """Proses OCR + chunking + indexing dokumen ke Qdrant."""
    try:
        logger.info(f"[API] doc-sync START → doc_id={sync_request.doc_id}, opd={sync_request.opd_name}, url={sync_request.file_url}")
        
        processing_result = process_document(
            doc_id=sync_request.doc_id,
            opd=sync_request.opd_name,
            file_url=sync_request.file_url,
            qdrant=qdrant,
            model=model_doc,
            lang=CONFIG.get("ocr", {}).get("lang", "id"),
            collection_name="document_bank"
        )
        
        if processing_result.get("status") != "ok":
            logger.warning(f"[API] doc-sync result NOT OK: {processing_result}")
            raise HTTPException(status_code=500, detail=processing_result)
        
        logger.info(f"[API] ✅ doc-sync FINISHED doc_id={sync_request.doc_id} | chunks={processing_result.get('total_chunks')}")
        return processing_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("[API] ❌ doc-sync error")
        raise HTTPException(status_code=500, detail=str(e))
