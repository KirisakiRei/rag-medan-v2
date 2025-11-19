from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

doc_search_router = APIRouter()
logger = logging.getLogger("doc_app")


class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5


@doc_search_router.post("/api/doc-search")
async def doc_search(req: DocSearchRequest, request: Request):
    """
    Endpoint untuk melakukan pencarian berbasis dokumen (hasil OCR).
    Jika USE_POST_SUMMARY=true di .env → hasil teratas diringkas.
    """
    try:
        # Import dependencies dari doc_app context
        from doc_app import qdrant, model_doc, embed_query
        from config import CONFIG
        from core.summarizer_utils import summarize_text
        
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
