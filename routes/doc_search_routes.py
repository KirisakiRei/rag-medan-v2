from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel
import logging

doc_search_router = APIRouter()
logger = logging.getLogger("doc_app")


class DocSearchRequest(BaseModel):
    query: str
    limit: int = 5


@doc_search_router.post("/api/doc-search")
async def doc_search(search_request: DocSearchRequest, request: Request):
    """
    Endpoint untuk melakukan pencarian berbasis dokumen (hasil OCR).
    Jika USE_POST_SUMMARY=true di .env → hasil teratas diringkas.
    """
    try:
        # Import di dalam fungsi untuk menghindari circular import
        from doc_app import qdrant, model_doc, embed_query
        from config import CONFIG
        from core.summarizer_utils import summarize_text
        
        request_source = request.headers.get("X-RAG-Source", "unknown")
        logger.info(f"[API] 🔍 doc-search query='{search_request.query}' limit={search_request.limit} | source={request_source}")

        # Embed query langsung
        query_vector = embed_query(model_doc, search_request.query)

        # Query ke Qdrant
        qdrant_hits = qdrant.query_points(
            collection_name="document_bank",
            query=query_vector,
            limit=search_request.limit
        )

        result_points = getattr(qdrant_hits, "points", None) or getattr(qdrant_hits, "result", None) or qdrant_hits
        search_results = []

        for hit in result_points:
            result_item = hit[0] if isinstance(hit, tuple) else hit
            result_payload = getattr(result_item, "payload", {}) or result_item.get("payload", {})
            result_score = getattr(result_item, "score", 0.0)

            search_results.append({
                "doc_id": result_payload.get("mysql_id"),
                "opd": result_payload.get("opd"),
                "filename": result_payload.get("filename"),
                "page_number": result_payload.get("page_number"),
                "chunk_index": result_payload.get("chunk_index"),
                "section": result_payload.get("section"),
                "summary": result_payload.get("summary"),
                "text": result_payload.get("text"),
                "score": float(result_score)
            })

        if not search_results:
            logger.info(f"[API] ⚠️ doc-search no results for query='{search_request.query}'")
            return {"status": "empty", "results": []}

        logger.info(f"[API] ✅ doc-search results={len(search_results)} hits | top_score={search_results[0]['score']:.3f}")

        # =====================================================
        # 🧠 Post Summarization (toggle via .env)
        # =====================================================
        use_post_summary = CONFIG.get("rag", {}).get("use_post_summary", False)
        post_summary_top_k = CONFIG.get("rag", {}).get("post_summary_top_k", 2)

        if use_post_summary:
            logger.info(f"[POST-SUM] Aktif → meringkas top {post_summary_top_k} hasil ...")
            top_ranked_results = sorted(search_results, key=lambda x: -x["score"])[:post_summary_top_k]
            combined_document_text = "\n\n".join(
                [result["text"] or "" for result in top_ranked_results if result.get("text")]
            )

            try:
                generated_summary = summarize_text(
                    f"Berdasarkan potongan dokumen berikut, jawab pertanyaan pengguna dengan ringkas dan informatif:\n\n{combined_document_text}",
                    max_sentences=5
                )
            except Exception as e:
                logger.warning(f"[POST-SUM] Gagal meringkas hasil: {e}")
                generated_summary = "Tidak dapat membuat ringkasan hasil."

            return {
                "status": "success",
                "mode": "post-summary",
                "query": search_request.query,
                "summary": generated_summary,
                "results": top_ranked_results
            }

        # =====================================================
        # 🚀 Mode Default (tanpa summarization)
        # =====================================================
        return {
            "status": "success",
            "mode": "direct",
            "query": search_request.query,
            "results": search_results
        }

    except Exception as e:
        logger.exception("❌ doc-search error")
        raise HTTPException(status_code=500, detail=str(e))
