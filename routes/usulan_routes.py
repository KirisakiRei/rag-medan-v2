from flask import Blueprint, request, jsonify
import time, logging, traceback
from qdrant_client.http import models
from core.filtering import ai_pre_filter_usulan, ai_relevance_usulan

usulan_bp = Blueprint("usulan_bp", __name__)
logger = logging.getLogger("app")
rag_summary_logger = logging.getLogger("rag.summary")


def error_response(error_type, message, detail=None, code=500):
    """Helper untuk membuat error response."""
    payload = {"status": "error", "error": {"type": error_type, "message": message}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


@usulan_bp.route("/api/sync-usulan", methods=["POST"])
def sync_usulan():
    """Endpoint untuk sinkronisasi data usulan_bank."""
    try:

        from app import model, qdrant
        
        request_data = request.json
        if not request_data or "action" not in request_data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)
        
        action = request_data["action"]
        content = request_data.get("content")
        collection = "usulan_bank"

        if action == "bulk_sync":
            if not isinstance(content, list):
                return error_response("ValidationError", "Content harus berupa list", code=400)
            
            points = []
            for item in content:
                vector = model.encode("passage: " + item["request_rag_name"]).tolist()
                point_id = str(item["request_rag_id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "request_id": item["request_id"],
                        "organization_id": item["organization_id"],
                        "request_name": item["request_name"],
                        "request_rag_name": item["request_rag_name"]
                    }
                })
            
            qdrant.upsert(collection_name=collection, points=points)
            qdrant.create_payload_index(
                collection_name=collection,
                field_name="request_rag_name",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                )
            )
            logger.info(f"[SYNC-USULAN] Sinkronisasi {len(points)} data ke {collection}")
            return jsonify({
                "status": "success",
                "message": f"{len(points)} data berhasil disinkronkan ke {collection}"
            }), 200

        elif action in ["add", "update"]:
            point_id = str(content["request_rag_id"])
            vector = model.encode("passage: " + content["request_rag_name"]).tolist()
            qdrant.upsert(
                collection_name=collection,
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "request_id": content["request_id"],
                        "organization_id": content["organization_id"],
                        "request_name": content["request_name"],
                        "request_rag_name": content["request_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-USULAN] Data {action} berhasil (ID={point_id})")
            return jsonify({"status": "success", "message": f"Data {action} berhasil"}), 200

        elif action == "delete":
            point_id = str(content["request_rag_id"])
            qdrant.delete(
                collection_name=collection,
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-USULAN] Data dihapus (ID={point_id})")
            return jsonify({"status": "success", "message": "Data berhasil dihapus"}), 200

        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[ERROR][sync_usulan] {str(e)}\n{error_traceback}")
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi usulan", detail=str(e))


@usulan_bp.route("/api/search-usulan", methods=["POST"])
def search_usulan():
    """Endpoint untuk pencarian usulan."""
    try:
        from app import model, qdrant
        
        start_time = time.time()
        request_data = request.json or {}
        user_request = (request_data.get("question") or "").strip()
        whatsapp_number = request_data.get("wa_number", "unknown")

        if not user_request:
            return jsonify({"status": "error", "message": "Field 'question' wajib diisi"}), 400

        logger.info("\n" + "=" * 60)
        logger.info("[USER-REQUEST] Request User : %s", user_request)
        logger.info("=" * 60)

        reformulation_start = time.time()
        reformulation_result = ai_pre_filter_usulan(user_request)
        reformulation_duration = time.time() - reformulation_start
        clean_request = reformulation_result.get("clean_request", user_request)

        embedding_start = time.time()
        query_vector = model.encode("query: " + clean_request).tolist()
        embedding_duration = time.time() - embedding_start

        qdrant_start = time.time()
        qdrant_results = qdrant.search(
            collection_name="usulan_bank",
            query_vector=query_vector,
            limit=5
        )
        qdrant_duration = time.time() - qdrant_start

        try:
            logger.info("\n" + "=" * 60)
            logger.info("[USULAN-SEARCH] Kandidat Hasil Pencarian Usulan")
            logger.info("-" * 60)

            if qdrant_results:
                for index, hit in enumerate(qdrant_results[:3], start=1):
                    request_name = (hit.payload.get("request_name") or "-").strip()
                    request_rag_name = (hit.payload.get("request_rag_name") or "-").strip()
                    request_id = hit.payload.get("request_id", "-")
                    organization_id = hit.payload.get("organization_id", "-")
                    dense_score = float(getattr(hit, "score", 0.0))

                    logger.info(
                        f"[{index}] Request Name     : {request_name}\n"
                        f"     Request Rag Name : {request_rag_name}\n"
                        f"     Request ID       : {request_id}\n"
                        f"     Org ID           : {organization_id}\n"
                        f"     Dense Score      : {dense_score:.3f}\n"
                        + "-" * 60
                    )
                logger.info("=" * 60 + "\n")
            else:
                logger.warning("[USULAN-SEARCH] Tidak ada hasil dari Qdrant.")
        except Exception as e:
            logger.error(f"[USULAN-SEARCH] Gagal mencetak hasil pencarian usulan: {e}")

        # ==================================================
        # ⚙️ SCORING
        # ==================================================
        accepted_results, rejected_results = [], []
        for hit in qdrant_results:
            dense_score = float(hit.score)
            final_score = round(dense_score, 3)
            acceptance_note, is_accepted = "-", False
            if dense_score >= 0.85:
                is_accepted, acceptance_note = True, "Data yang Relevan Ditemukan"

            result_item = {
                "request_id": hit.payload.get("request_id"),
                "organization_id": hit.payload.get("organization_id"),
                "request_name": hit.payload.get("request_name"),
                "request_rag_name": hit.payload.get("request_rag_name"),
                "dense_score": dense_score,
                "final_score": final_score,
                "note": acceptance_note
            }
            (accepted_results if is_accepted else rejected_results).append(result_item)

        accepted_results = sorted(accepted_results, key=lambda x: x["final_score"], reverse=True)
        rejected_results = sorted(rejected_results, key=lambda x: x["final_score"], reverse=True)

        # ==================================================
        # 🤖 AI TOPIC RELEVANCE CHECK
        # ==================================================
        if qdrant_results:
            top_rag_name = qdrant_results[0].payload.get("request_rag_name", "-")
            topic_check_result = ai_relevance_usulan(user_request, top_rag_name)
        else:
            topic_check_result = {"relevant": True, "reason": "Tidak ada hasil RAG"}

        if not topic_check_result.get("relevant", True):
            total_duration = time.time() - start_time
            logger.info(f"[AI-TOPIC-USULAN] Topik tidak relevan | Reason: {topic_check_result.get('reason')}")

            # ⭐ TAMBAHAN — LOG OUTPUT YANG AKAN DIKIRIM KE WABOT (kosong)
            logger.info("\n" + "=" * 60)
            logger.info("[USULAN-POST] Output akan dikirim ke WABOT: '-' (Tidak relevan)")
            logger.info("=" * 60)

            # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
            logger.info(f"[REQUEST] Total waktu permintaan: {total_duration:.3f} detik")
            logger.info("=" * 60 + "\n")

            rag_summary_logger.info(
                f"\n{'='*60}\n[USULAN TOPIC CHECK]\nUser: {user_request}\nTopik RAG: {top_rag_name}\n"
                f"Relevan: {topic_check_result.get('relevant')} | Reason: {topic_check_result.get('reason')}\n{'='*60}\n"
            )
            return jsonify({
                "status": "low_confidence",
                "message": "Topik tidak relevan dengan pertanyaan pengguna",
                "reason": topic_check_result.get("reason", "-"),
                "data": {"similar_questions": []},
                "timing": {"total_sec": round(total_duration, 3)}
            }), 200

        # ==================================================
        # ⭐ TAMBAHAN — LOG OUTPUT AKHIR YANG DIKIRIM KE WABOT
        # ==================================================
        if accepted_results:
            final_usulan_output = accepted_results[0]["request_rag_name"]
        else:
            final_usulan_output = "-"

        logger.info("\n" + "=" * 60)
        logger.info(f"[USULAN-POST] Output akan dikirim ke WABOT: '{final_usulan_output}'")
        logger.info("=" * 60)

        # ==================================================
        # ⏱️ WAKTU TOTAL & PAYLOAD
        # ==================================================
        total_duration = time.time() - start_time
        response_payload = {
            "status": "success" if accepted_results else "low_confidence",
            "message": "Hasil ditemukan" if accepted_results else "Tidak ada hasil cukup relevan",
            "data": {
                "similar_questions": accepted_results if accepted_results else rejected_results,
                "metadata": {
                    "wa_number": whatsapp_number,
                    "user_question": user_request,
                    "final_score_top": (accepted_results[0]["final_score"] if accepted_results else "-")
                }
            },
            "timing": {
                "reform_sec": round(reformulation_duration, 3),
                "embedding_sec": round(embedding_duration, 3),
                "qdrant_sec": round(qdrant_duration, 3),
                "total_sec": round(total_duration, 3)
            }
        }

        # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
        logger.info(f"[REQUEST] Total waktu permintaan: {total_duration:.3f} detik")
        logger.info("=" * 60 + "\n")

        # ==================================================
        # 🧾 LOG RINGKASAN
        # ==================================================
        try:
            summary_lines = [
                "\n" + "=" * 60,
                "[USULAN SEARCH SESSION]",
                f"Pertanyaan User: {user_request}",
                f"Hasil Pencarian: {len(qdrant_results)} kandidat ditemukan",
                f"Hasil Cek Topik: {topic_check_result.get('relevant')} | Reason: {topic_check_result.get('reason')}",
            ]

            for index, result in enumerate(accepted_results[:3], start=1):
                summary_lines.append(
                    f"{index}. {result['request_rag_name']} | ReqID={result['request_id']} | Dense={result['dense_score']:.3f}"
                )

            result_ids = [r.get("request_id") for r in accepted_results[:3] if r.get("request_id")]
            summary_lines.append(f"Request ID yang dikembalikan: {result_ids}")
            summary_lines.append(
                f"Total waktu proses: {total_duration:.3f} detik "
                f"(Reform={reformulation_duration:.3f}s | Emb={embedding_duration:.3f}s | Qdrant={qdrant_duration:.3f}s)"
            )
            summary_lines.append("=" * 60 + "\n")

            rag_summary_logger.info("\n".join(summary_lines))
        except Exception as e:
            logger.warning(f"[LOGGING ERROR] Gagal mencetak ringkasan USULAN: {e}")

        return jsonify(response_payload), 200

    except Exception as e:
        error_traceback = traceback.format_exc()
        logger.error(f"[ERROR][search_usulan] {str(e)}\n{error_traceback}")
        return error_response("ServerError", "Kesalahan internal saat pencarian usulan", detail=str(e))
