# ============================================================
# ✅ RAG Document API — Standalone dengan Post-Summarization Toggle
# ============================================================
import logging, sys, os
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
import uvicorn
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
from config import CONFIG
from core.document_pipeline import process_document
from core.summarizer_utils import summarize_text  # 🔹 pakai summarizer LLM

# ============================================================
# 🔹 Setup Logging
# ============================================================
LOG_FILE = "./logs/rag-doc.log"

os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)

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
# 🔹 Model & Qdrant setup
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
# 🔹 Middleware — Logging sumber request
# ============================================================
@app.middleware("http")
async def add_source_logger(request: Request, call_next):
    source = request.headers.get("X-RAG-Source", "unknown")
    path = request.url.path
    if "doc-search" in path:
        logger.info(f"[MIDDLEWARE] Request from {source} → {path}")
    response = await call_next(request)
    return response

# ============================================================
# 🔹 Request Models
# ============================================================
class DocSyncRequest(BaseModel):
    doc_id: str
    opd_name: str | None = None
    file_url: str

class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5

# ============================================================
# 🔹 Document Sync API
# ============================================================
@app.post("/api/doc-sync")
async def doc_sync(req: DocSyncRequest):
    """Proses OCR + chunking + indexing dokumen ke Qdrant."""
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
# 🔹 Document Search API — dgn Post-Summarization Toggle
# ============================================================
@app.post("/api/doc-search")
async def doc_search(req: DocSearchRequest, request: Request):
    """
    Endpoint untuk melakukan pencarian berbasis dokumen (hasil OCR).
    Jika USE_POST_SUMMARY=true di .env → hasil teratas diringkas.
    """
    try:
        source = request.headers.get("X-RAG-Source", "unknown")
        logger.info(f"[API] 🔍 doc-search query='{req.query}' limit={req.limit} | source={source}")

        # Embed query langsung
        vec = embed_query(model_doc, req.query)

        # Query ke Qdrant
        hits = qdrant.query_points(
            collection_name="document_bank",
            query=vec,
            limit=req.limit
        )

        points = getattr(hits, "points", None) or getattr(hits, "result", None) or hits
        results = []

        for h in points:
            item = h[0] if isinstance(h, tuple) else h
            payload = getattr(item, "payload", {}) or item.get("payload", {})
            score = getattr(item, "score", 0.0)

            results.append({
                "doc_id": payload.get("mysql_id"),
                "opd": payload.get("opd"),
                "filename": payload.get("filename"),
                "page_number": payload.get("page_number"),
                "chunk_index": payload.get("chunk_index"),
                "section": payload.get("section"),
                "summary": payload.get("summary"),
                "text": payload.get("text"),
                "score": float(score)
            })

        if not results:
            logger.info(f"[API] ⚠️ doc-search no results for query='{req.query}'")
            return {"status": "empty", "results": []}

        logger.info(f"[API] ✅ doc-search results={len(results)} hits | top_score={results[0]['score']:.3f}")

        # =====================================================
        # 🧠 Post Summarization (toggle via .env)
        # =====================================================
        use_post_summary = CONFIG.get("rag", {}).get("use_post_summary", False)
        top_k = CONFIG.get("rag", {}).get("post_summary_top_k", 2)

        if use_post_summary:
            logger.info(f"[POST-SUM] Aktif → meringkas top {top_k} hasil ...")
            top_results = sorted(results, key=lambda x: -x["score"])[:top_k]
            combined_text = "\n\n".join(
                [r["text"] or "" for r in top_results if r.get("text")]
            )

            try:
                summary = summarize_text(
                    f"Berdasarkan potongan dokumen berikut, jawab pertanyaan pengguna dengan ringkas dan informatif:\n\n{combined_text}",
                    max_sentences=5
                )
            except Exception as e:
                logger.warning(f"[POST-SUM] Gagal meringkas hasil: {e}")
                summary = "Tidak dapat membuat ringkasan hasil."

            return {
                "status": "success",
                "mode": "post-summary",
                "query": req.query,
                "summary": summary,
                "results": top_results
            }

        # =====================================================
        # 🚀 Mode Default (tanpa summarization)
        # =====================================================
        return {
            "status": "success",
            "mode": "direct",
            "query": req.query,
            "results": results
        }

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
        log_config=None
    )
