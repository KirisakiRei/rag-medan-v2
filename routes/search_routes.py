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


def error_response(error_type, message, detail=None, code=500):
    """Helper untuk membuat error response."""
    payload = {"status": "error", "error": {"type": error_type, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


@search_bp.route("/api/search", methods=["POST"])
def search():
    """Endpoint untuk pencarian di RAG text dengan fallback ke RAG dokumen."""
    try:
        # Import model dan qdrant dari app context
        from app import model, qdrant
        
        start_time = time.time()
        request_data = request.json or {}
        user_question = (request_data.get("question") or "").strip()
        whatsapp_number = request_data.get("wa_number", "unknown")

        if not user_question:
            return jsonify({"status": "error", "message": "Field 'question' wajib diisi"}), 400

        # ⭐ TAMBAHAN — LOG PERTANYAAN USER
        logger.info("\n" + "=" * 60)
        logger.info("[USER-QUESTION] Pertanyaan User : %s", user_question)
        logger.info("=" * 60)

        # ==================================================
        # 🧩 AI FILTER (PRE)
        # ==================================================
        pre_filter_start = time.time()
        pre_filter_result = ai_pre_filter(user_question)
        pre_filter_duration = time.time() - pre_filter_start

        if not pre_filter_result.get("valid", True):
            total_duration = time.time() - start_time
            return jsonify({
                "status": "low_confidence",
                "message": pre_filter_result.get("reason", "Pertanyaan tidak relevan"),
                "data": {
                    "similar_questions": [],
                    "metadata": {
                        "wa_number": whatsapp_number,
                        "original_question": user_question,
                        "final_question": "-",
                        "category": "-",
                        "ai_reason": pre_filter_result.get("reason", "-"),
                        "ai_reformulated": "-",
                        "final_score_top": "-"
                    }
                },
                "timing": {
                    "ai_domain_sec": round(pre_filter_duration, 3),
                    "ai_relevance_sec": 0.0,
                    "embedding_sec": 0.0,
                    "qdrant_sec": 0.0,
                    "total_sec": round(total_duration, 3)
                }
            }), 200

        # ==================================================
        # 🧩 EMBEDDING & CARI KE QDRANT
        # ==================================================
        normalized_question = normalize_text(clean_location_terms(pre_filter_result.get("clean_question", user_question)))
        detected_category = detect_category(normalized_question)
        category_id = detected_category["id"] if detected_category else None

        embedding_start = time.time()
        query_vector = model.encode("query: " + normalized_question).tolist()
        embedding_duration = time.time() - embedding_start

        qdrant_start = time.time()
        category_filter = models.Filter(must=[
            models.FieldCondition(
                key="category_id",
                match=models.MatchValue(value=category_id)
            )
        ]) if category_id else None

        qdrant_results = qdrant.search(
            collection_name="knowledge_bank",
            query_vector=query_vector,
            limit=5,
            query_filter=category_filter
        )
        qdrant_duration = time.time() - qdrant_start

        # ==================================================
        # 🧾 LOG HASIL QDRANT
        # ==================================================
        try:
            if qdrant_results:
                logger.info("\n" + "=" * 60)
                logger.info("[RAG-SEARCH] Kandidat Hasil Pencarian Awal")
                logger.info("-" * 60)

                for index, hit in enumerate(qdrant_results[:3], start=1):
                    rag_question = (hit.payload.get("question_rag_name") or "-").strip()
                    dense_score = float(getattr(hit, "score", 0.0))
                    answer_id = safe_parse_answer_id(hit.payload.get("answer_id"))
                    category_id_hit = hit.payload.get("category_id", "-")

                    overlap_score = keyword_overlap(normalized_question, rag_question)
                    final_score = round((0.65 * dense_score) + (0.35 * overlap_score), 3)

                    logger.info(
                        f"[{index}] Question : {rag_question}\n"
                        f"     DenseScore  : {dense_score:.3f}\n"
                        f"     OverlapScore: {overlap_score:.3f}\n"
                        f"     FinalScore  : {final_score:.3f}\n"
                        f"     CategoryID  : {category_id_hit}\n"
                        f"     AnswerID    : {answer_id}\n"
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
        relevance_check_start = time.time()
        relevance_result = {}
        if qdrant_results:
            relevance_result = ai_check_relevance(user_question, qdrant_results[0].payload["question_rag_name"])
        relevance_check_duration = time.time() - relevance_check_start

        # ==================================================
        # ⚙️ SCORING
        # ==================================================
        accepted_results, rejected_results = [], []
        for hit in qdrant_results:
            dense_score = float(hit.score)
            overlap_score = keyword_overlap(normalized_question, hit.payload["question_rag_name"])
            final_score = round((0.65 * dense_score) + (0.35 * overlap_score), 3)

            acceptance_note, is_accepted = "-", False
            if dense_score >= 0.90:
                is_accepted, acceptance_note = True, "auto_accepted_by_dense"
            elif 0.86 <= dense_score <= 0.89 and overlap_score >= 0.25:
                is_accepted, acceptance_note = True, "accepted_by_overlap"

            try:
                if not is_accepted and dense_score >= 0.83 and overlap_score >= 0.15 and relevance_result.get("relevant", False):
                    is_accepted, acceptance_note = True, "accepted_by_ai_relevance"
            except Exception:
                pass

            result_item = {
                "question": hit.payload["question"],
                "question_rag_name": hit.payload["question_rag_name"],
                "answer_id": safe_parse_answer_id(hit.payload.get("answer_id")),
                "answer_doc": "",
                "category_id": hit.payload.get("category_id"),
                "dense_score": dense_score,
                "overlap_score": overlap_score,
                "final_score": final_score,
                "note": acceptance_note
            }
            (accepted_results if is_accepted else rejected_results).append(result_item)

        accepted_results = sorted(accepted_results, key=lambda x: x["final_score"], reverse=True)
        rejected_results = sorted(rejected_results, key=lambda x: x["final_score"], reverse=True)

        # ==================================================
        # 🚫 FILTER KETIKA RELEVANCE = FALSE
        # ==================================================
        is_question_relevant = relevance_result.get("relevant", True)
        if not is_question_relevant:
            logger.info("[AI-POST] Pertanyaan dinilai TIDAK relevan oleh model relevance-check.")
            accepted_results = []

        # ⭐ TAMBAHAN — LOG OUTPUT YANG AKAN DIKIRIM KE WABOT
        if accepted_results:
            final_rag_output = accepted_results[0]["question_rag_name"]
        else:
            final_rag_output = "-"

        logger.info("\n" + "=" * 60)
        logger.info(f"[AI-POST] Output akan dikirim ke WABOT: '{final_rag_output}'")
        logger.info("=" * 60)

        # ==================================================
        # 📦 RESPONSE PAYLOAD
        # ==================================================
        total_duration = time.time() - start_time
        response_payload = {
            "status": "success" if accepted_results else "low_confidence",
            "message": "Hasil ditemukan" if accepted_results else "Tidak ada hasil cukup relevan",
            "source": "text",
            "data": {
                "similar_questions": accepted_results if accepted_results else rejected_results,
                "metadata": {
                    "wa_number": whatsapp_number,
                    "original_question": user_question,
                    "final_question": normalized_question,
                    "category": (detected_category["name"] if detected_category else "Global"),
                    "ai_reason": relevance_result.get("reason", "-") if relevance_result else "-",
                    "ai_reformulated": relevance_result.get("reformulated_question", "-") if relevance_result else "-",
                    "final_score_top": (accepted_results[0]["final_score"] if accepted_results else "-")
                }
            },
            "timing": {
                "ai_domain_sec": round(pre_filter_duration, 3),
                "ai_relevance_sec": round(relevance_check_duration, 3),
                "embedding_sec": round(embedding_duration, 3),
                "qdrant_sec": round(qdrant_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }

        # ==================================================
        # 🔄 FALLBACK KE RAG DOKUMEN
        # Hanya jika: (1) Tidak ada hasil dari Qdrant, ATAU (2) AI Relevance = False
        # ==================================================
        try:
            should_fallback_to_document = (
                len(qdrant_results) == 0 or
                not is_question_relevant
            )

            if should_fallback_to_document:
                logger.info("[FALLBACK] Tidak ada hasil dari RAG text atau tidak relevan → mencoba ke RAG dokumen")
                doc_api_url = f"{CONFIG['doc_api']['base_url']}/api/doc-search"

                try:
                    doc_response = requests.post(
                        doc_api_url,
                        json={"query": user_question, "limit": 3},
                        headers={"X-RAG-Source": "text-fallback"},
                        timeout=12
                    )

                    if doc_response.status_code == 200:
                        doc_response_data = doc_response.json()
                        if doc_response_data.get("status") == "success" and doc_response_data.get("results"):
                            top_document = doc_response_data["results"][0]
                            document_text = top_document.get("text", "")
                            
                            # ==================================================
                            # 🧠 AI RELEVANCE CHECK UNTUK DOKUMEN
                            # ==================================================
                            logger.info("[FALLBACK] Mengecek relevansi hasil dokumen dengan AI...")
                            doc_relevance_start = time.time()
                            doc_relevance_result = ai_check_relevance(user_question, document_text)
                            doc_relevance_duration = time.time() - doc_relevance_start
                            
                            is_document_relevant = doc_relevance_result.get("relevant", False)
                            logger.info(f"[FALLBACK] AI Relevance dokumen: {is_document_relevant} | Reason: {doc_relevance_result.get('reason', '-')}")
                            
                            if is_document_relevant:
                                # Format payload dengan answer_doc untuk hasil dokumen
                                response_payload = {
                                    "status": "success",
                                    "message": "Hasil ditemukan dari dokumen",
                                    "source": "document",
                                    "data": {
                                        "similar_questions": [{
                                            "question": "-",
                                            "question_rag_name": "-",
                                            "answer_id": None,
                                            "answer_doc": document_text,
                                            "category_id": None,
                                            "dense_score": round(top_document.get("score", 0.0), 3),
                                            "overlap_score": 0.0,
                                            "final_score": round(top_document.get("score", 0.0), 3),
                                            "note": "from_document_rag"
                                        }],
                                        "metadata": {
                                            "wa_number": whatsapp_number,
                                            "original_question": user_question,
                                            "final_question": normalized_question,
                                            "category": "Dokumen",
                                            "ai_reason": doc_relevance_result.get("reason", "-"),
                                            "ai_reformulated": doc_relevance_result.get("reformulated_question", "-"),
                                            "final_score_top": round(top_document.get("score", 0.0), 3),
                                            "document_info": {
                                                "filename": top_document.get("filename", "-"),
                                                "page_number": top_document.get("page_number", "-"),
                                                "opd": top_document.get("opd", "-")
                                            }
                                        }
                                    },
                                    "timing": {
                                        **response_payload["timing"],
                                        "ai_relevance_doc_sec": round(doc_relevance_duration, 3)
                                    }
                                }
                                logger.info("[FALLBACK] ✅ Jawaban relevan dan ditemukan di RAG dokumen")
                                rag_summary_logger.info(
                                    f"[RAG FALLBACK] {user_question} → Dokumen: {top_document.get('filename')} (score={top_document.get('score'):.3f}, relevant=True)"
                                )
                            else:
                                # Dokumen tidak relevan
                                logger.info("[FALLBACK] ❌ Hasil dokumen tidak relevan dengan pertanyaan user")
                                response_payload["source"] = "none"
                                response_payload["message"] = "Tidak ada hasil relevan ditemukan"
                                rag_summary_logger.info(
                                    f"[RAG FALLBACK] {user_question} → Dokumen tidak relevan (reason={doc_relevance_result.get('reason', '-')})"
                                )
                        else:
                            logger.info("[FALLBACK] Tidak ada hasil relevan di RAG dokumen.")
                            response_payload["source"] = "none"
                    else:
                        logger.warning(f"[FALLBACK] Gagal menghubungi RAG dokumen: HTTP {doc_response.status_code}")
                        response_payload["source"] = "none"

                except requests.exceptions.RequestException as e:
                    logger.error(f"[FALLBACK] Error saat memanggil RAG dokumen: {e}")
                    response_payload["source"] = "none"
            else:
                logger.info("[FALLBACK] ✅ Hasil ditemukan di RAG text, tidak perlu fallback ke dokumen")

        except Exception as e:
            logger.error(f"[FALLBACK ERROR] {e}")
            response_payload["source"] = "none"

        # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
        logger.info(f"[REQUEST] Total waktu permintaan: {total_duration:.3f} detik")
        logger.info("=" * 60 + "\n")

        return jsonify(response_payload), 200

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[ERROR][search] {str(e)}\n{error_traceback}")
        return error_response("ServerError", "Kesalahan internal", detail=str(e))
