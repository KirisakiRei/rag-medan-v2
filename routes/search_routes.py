from flask import Blueprint, request, jsonify
import time, logging, traceback, requests
from qdrant_client.http import models
from config import CONFIG
from core.utils import (
    detect_category,
    normalize_text,
    clean_location_terms,
    keyword_overlap,
    safe_parse_answer_id,
)
from core.filtering import ai_pre_filter, ai_check_relevance

search_bp = Blueprint("search_bp", __name__)
logger = logging.getLogger("app")
rag_summary_logger = logging.getLogger("rag.summary")


def error_response(t, msg, detail=None, code=500):
    """Helper untuk membuat error response."""
    payload = {"status": "error", "error": {"type": t, "message": msg}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


@search_bp.route("/api/search", methods=["POST"])
def search():
    """Endpoint untuk pencarian di RAG text dengan fallback ke RAG dokumen."""
    try:
        # Import model dan qdrant dari app context
        from app import model, qdrant
        
        t0 = time.time()
        data = request.json or {}
        user_q = (data.get("question") or "").strip()
        wa = data.get("wa_number", "unknown")

        if not user_q:
            return jsonify({"status": "error", "message": "Field 'question' wajib diisi"}), 400

        # ⭐ TAMBAHAN — LOG PERTANYAAN USER
        logger.info("\n" + "=" * 60)
        logger.info("[USER-QUESTION] Pertanyaan User : %s", user_q)
        logger.info("=" * 60)

        # ==================================================
        # 🧩 AI FILTER (PRE)
        # ==================================================
        t_pre = time.time()
        pre = ai_pre_filter(user_q)
        t_pre_time = time.time() - t_pre

        if not pre.get("valid", True):
            total_time = time.time() - t0
            return jsonify({
                "status": "low_confidence",
                "message": pre.get("reason", "Pertanyaan tidak relevan"),
                "data": {
                    "similar_questions": [],
                    "metadata": {
                        "wa_number": wa,
                        "original_question": user_q,
                        "final_question": "-",
                        "category": "-",
                        "ai_reason": pre.get("reason", "-"),
                        "ai_reformulated": "-",
                        "final_score_top": "-"
                    }
                },
                "timing": {
                    "ai_domain_sec": round(t_pre_time, 3),
                    "ai_relevance_sec": 0.0,
                    "embedding_sec": 0.0,
                    "qdrant_sec": 0.0,
                    "total_sec": round(total_time, 3)
                }
            }), 200

        # ==================================================
        # 🧩 EMBEDDING & CARI KE QDRANT
        # ==================================================
        question = normalize_text(clean_location_terms(pre.get("clean_question", user_q)))
        category = detect_category(question)
        cat_id = category["id"] if category else None

        t_emb = time.time()
        qvec = model.encode("query: " + question).tolist()
        emb_time = time.time() - t_emb

        t_qd = time.time()
        filt = models.Filter(must=[
            models.FieldCondition(
                key="category_id",
                match=models.MatchValue(value=cat_id)
            )
        ]) if cat_id else None

        dense_hits = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=qvec,
            limit=5,
            query_filter=filt
        )
        qd_time = time.time() - t_qd

        # ==================================================
        # 🧾 LOG HASIL QDRANT
        # ==================================================
        try:
            if dense_hits:
                logger.info("\n" + "=" * 60)
                logger.info("[RAG-SEARCH] Kandidat Hasil Pencarian Awal")
                logger.info("-" * 60)

                for idx, h in enumerate(dense_hits[:3], start=1):
                    q = (h.payload.get("question_rag_name") or "-").strip()
                    s_dense = float(getattr(h, "score", 0.0))
                    a = safe_parse_answer_id(h.payload.get("answer_id"))
                    c = h.payload.get("category_id", "-")

                    overlap = keyword_overlap(question, q)
                    final_score = round((0.65 * s_dense) + (0.35 * overlap), 3)

                    logger.info(
                        f"[{idx}] Question : {q}\n"
                        f"     DenseScore  : {s_dense:.3f}\n"
                        f"     OverlapScore: {overlap:.3f}\n"
                        f"     FinalScore  : {final_score:.3f}\n"
                        f"     CategoryID  : {c}\n"
                        f"     AnswerID    : {a}\n"
                        + "-" * 60
                    )
                logger.info("=" * 60 + "\n")
            else:
                logger.warning("[RAG-SEARCH] Tidak ada hasil dari Qdrant.")
        except Exception as e:
            logger.error(f"[RAG-SEARCH] Gagal mencetak hasil pencarian awal: {e}")

        # ==================================================
        # 🧠 AI RELEVANCE CHECK
        # ==================================================
        t_post = time.time()
        relevance = {}
        if dense_hits:
            relevance = ai_check_relevance(user_q, dense_hits[0].payload["question_rag_name"])
        t_post_time = time.time() - t_post

        # ==================================================
        # ⚙️ SCORING
        # ==================================================
        results, rejected = [], []
        for h in dense_hits:
            dense = float(h.score)
            overlap = keyword_overlap(question, h.payload["question_rag_name"])
            final_score = round((0.65 * dense) + (0.35 * overlap), 3)

            note, accepted = "-", False
            if dense >= 0.90:
                accepted, note = True, "auto_accepted_by_dense"
            elif 0.86 <= dense <= 0.89 and overlap >= 0.25:
                accepted, note = True, "accepted_by_overlap"

            try:
                if not accepted and dense >= 0.83 and overlap >= 0.15 and relevance.get("relevant", False):
                    accepted, note = True, "accepted_by_ai_relevance"
            except Exception:
                pass

            item = {
                "question": h.payload["question"],
                "question_rag_name": h.payload["question_rag_name"],
                "answer_id": safe_parse_answer_id(h.payload.get("answer_id")),
                "answer_doc": "",
                "category_id": h.payload.get("category_id"),
                "dense_score": dense,
                "overlap_score": overlap,
                "final_score": final_score,
                "note": note
            }
            (results if accepted else rejected).append(item)

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        rejected = sorted(rejected, key=lambda x: x["final_score"], reverse=True)

        # ==================================================
        # 🚫 FILTER KETIKA RELEVANCE = FALSE
        # ==================================================
        is_relevant = relevance.get("relevant", True)
        if not is_relevant:
            logger.info("[AI-POST] Pertanyaan dinilai TIDAK relevan oleh model relevance-check.")
            results = []

        # ⭐ TAMBAHAN — LOG OUTPUT YANG AKAN DIKIRIM KE WABOT
        if results:
            final_rag = results[0]["question_rag_name"]
        else:
            final_rag = "-"

        logger.info("\n" + "=" * 60)
        logger.info(f"[AI-POST] Output akan dikirim ke WABOT: '{final_rag}'")
        logger.info("=" * 60)

        # ==================================================
        # 📦 RESPONSE PAYLOAD
        # ==================================================
        total_time = time.time() - t0
        payload = {
            "status": "success" if results else "low_confidence",
            "message": "Hasil ditemukan" if results else "Tidak ada hasil cukup relevan",
            "source": "text",
            "data": {
                "similar_questions": results if results else rejected,
                "metadata": {
                    "wa_number": wa,
                    "original_question": user_q,
                    "final_question": question,
                    "category": (category["name"] if category else "Global"),
                    "ai_reason": relevance.get("reason", "-") if relevance else "-",
                    "ai_reformulated": relevance.get("reformulated_question", "-") if relevance else "-",
                    "final_score_top": (results[0]["final_score"] if results else "-")
                }
            },
            "timing": {
                "ai_domain_sec": round(t_pre_time, 3),
                "ai_relevance_sec": round(t_post_time, 3),
                "embedding_sec": round(emb_time, 3),
                "qdrant_sec": round(qd_time, 3),
                "total_sec": round(total_time, 3)
            }
        }

        # ==================================================
        # 🔄 FALLBACK KE RAG DOKUMEN
        # Hanya jika: (1) Tidak ada hasil dari Qdrant, ATAU (2) AI Relevance = False
        # ==================================================
        try:
            should_fallback = (
                len(dense_hits) == 0 or  # Tidak ada hasil dari Qdrant
                not is_relevant           # AI Relevance menganggap tidak relevan
            )

            if should_fallback:
                logger.info("[FALLBACK] Tidak ada hasil dari RAG text atau tidak relevan → mencoba ke RAG dokumen")
                doc_api_url = f"{CONFIG['doc_api']['base_url']}/api/doc-search"

                try:
                    doc_response = requests.post(
                        doc_api_url,
                        json={"query": user_q, "limit": 3},
                        headers={"X-RAG-Source": "text-fallback"},
                        timeout=12
                    )

                    if doc_response.status_code == 200:
                        doc_data = doc_response.json()
                        if doc_data.get("status") == "success" and doc_data.get("results"):
                            top = doc_data["results"][0]
                            doc_text = top.get("text", "")
                            
                            # ==================================================
                            # 🧠 AI RELEVANCE CHECK UNTUK DOKUMEN
                            # ==================================================
                            logger.info("[FALLBACK] Mengecek relevansi hasil dokumen dengan AI...")
                            t_doc_relevance = time.time()
                            doc_relevance = ai_check_relevance(user_q, doc_text)
                            t_doc_relevance_time = time.time() - t_doc_relevance
                            
                            is_doc_relevant = doc_relevance.get("relevant", False)
                            logger.info(f"[FALLBACK] AI Relevance dokumen: {is_doc_relevant} | Reason: {doc_relevance.get('reason', '-')}")
                            
                            if is_doc_relevant:
                                # Format payload dengan answer_doc untuk hasil dokumen
                                payload = {
                                    "status": "success",
                                    "message": "Hasil ditemukan dari dokumen",
                                    "source": "document",
                                    "data": {
                                        "similar_questions": [{
                                            "question": "-",
                                            "question_rag_name": "-",
                                            "answer_id": None,
                                            "answer_doc": doc_text,
                                            "category_id": None,
                                            "dense_score": round(top.get("score", 0.0), 3),
                                            "overlap_score": 0.0,
                                            "final_score": round(top.get("score", 0.0), 3),
                                            "note": "from_document_rag"
                                        }],
                                        "metadata": {
                                            "wa_number": wa,
                                            "original_question": user_q,
                                            "final_question": question,
                                            "category": "Dokumen",
                                            "ai_reason": doc_relevance.get("reason", "-"),
                                            "ai_reformulated": doc_relevance.get("reformulated_question", "-"),
                                            "final_score_top": round(top.get("score", 0.0), 3),
                                            "document_info": {
                                                "filename": top.get("filename", "-"),
                                                "page_number": top.get("page_number", "-"),
                                                "opd": top.get("opd", "-")
                                            }
                                        }
                                    },
                                    "timing": {
                                        **payload["timing"],
                                        "ai_relevance_doc_sec": round(t_doc_relevance_time, 3)
                                    }
                                }
                                logger.info("[FALLBACK] ✅ Jawaban relevan dan ditemukan di RAG dokumen")
                                rag_summary_logger.info(
                                    f"[RAG FALLBACK] {user_q} → Dokumen: {top.get('filename')} (score={top.get('score'):.3f}, relevant=True)"
                                )
                            else:
                                # Dokumen tidak relevan
                                logger.info("[FALLBACK] ❌ Hasil dokumen tidak relevan dengan pertanyaan user")
                                payload["source"] = "none"
                                payload["message"] = "Tidak ada hasil relevan ditemukan"
                                rag_summary_logger.info(
                                    f"[RAG FALLBACK] {user_q} → Dokumen tidak relevan (reason={doc_relevance.get('reason', '-')})"
                                )
                        else:
                            logger.info("[FALLBACK] Tidak ada hasil relevan di RAG dokumen.")
                            payload["source"] = "none"
                    else:
                        logger.warning(f"[FALLBACK] Gagal menghubungi RAG dokumen: HTTP {doc_response.status_code}")
                        payload["source"] = "none"

                except requests.exceptions.RequestException as e:
                    logger.error(f"[FALLBACK] Error saat memanggil RAG dokumen: {e}")
                    payload["source"] = "none"
            else:
                logger.info("[FALLBACK] ✅ Hasil ditemukan di RAG text, tidak perlu fallback ke dokumen")

        except Exception as e:
            logger.error(f"[FALLBACK ERROR] {e}")
            payload["source"] = "none"

        # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
        logger.info(f"[REQUEST] Total waktu permintaan: {total_time:.3f} detik")
        logger.info("=" * 60 + "\n")

        return jsonify(payload), 200

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][search] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal", detail=str(e))
