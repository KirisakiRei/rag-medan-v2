from flask import Blueprint, request, jsonify
import time, logging, traceback
from qdrant_client.http import models
from core.filtering import ai_pre_filter_usulan, ai_relevance_usulan

usulan_bp = Blueprint("usulan_bp", __name__)
logger = logging.getLogger("app")
rag_summary_logger = logging.getLogger("rag.summary")


def error_response(t, msg, detail=None, code=500):
    """Helper untuk membuat error response."""
    payload = {"status": "error", "error": {"type": t, "message": msg}}
    if detail:
        payload["error"]["detail"] = detail
    return jsonify(payload), code


@usulan_bp.route("/api/sync-usulan", methods=["POST"])
def sync_usulan():
    """Endpoint untuk sinkronisasi data usulan_bank."""
    try:
        # Import model dan qdrant dari app context
        from app import model, qdrant
        
        data = request.json
        if not data or "action" not in data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)
        
        action = data["action"]
        content = data.get("content")
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
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][sync_usulan] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi usulan", detail=str(e))


@usulan_bp.route("/api/search-usulan", methods=["POST"])
def search_usulan():
    """Endpoint untuk pencarian usulan."""
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
        logger.info("[USER-REQUEST] Request User : %s", user_q)
        logger.info("=" * 60)

        # ==================================================
        # 🧩 AI PRE-FILTER (REFORMULASI INPUT)
        # ==================================================
        t_ref = time.time()
        reform = ai_pre_filter_usulan(user_q)
        t_ref_time = time.time() - t_ref
        clean_q = reform.get("clean_request", user_q)

        # ==================================================
        # 🧠 EMBEDDING
        # ==================================================
        t_emb = time.time()
        qvec = model.encode("query: " + clean_q).tolist()
        emb_time = time.time() - t_emb

        # ==================================================
        # 🗃️ QDRANT SEARCH
        # ==================================================
        t_qd = time.time()
        dense_hits = qdrant.search(
            collection_name="usulan_bank",
            query_vector=qvec,
            limit=5
        )
        qd_time = time.time() - t_qd

        # ==================================================
        # 🧾 LOG HASIL PENCARIAN USULAN
        # ==================================================
        try:
            logger.info("\n" + "=" * 60)
            logger.info("[USULAN-SEARCH] Kandidat Hasil Pencarian Usulan")
            logger.info("-" * 60)

            if dense_hits:
                for idx, h in enumerate(dense_hits[:3], start=1):
                    req_name = (h.payload.get("request_name") or "-").strip()
                    req_rag_name = (h.payload.get("request_rag_name") or "-").strip()
                    req_id = h.payload.get("request_id", "-")
                    org_id = h.payload.get("organization_id", "-")
                    dense = float(getattr(h, "score", 0.0))

                    logger.info(
                        f"[{idx}] Request Name     : {req_name}\n"
                        f"     Request Rag Name : {req_rag_name}\n"
                        f"     Request ID       : {req_id}\n"
                        f"     Org ID           : {org_id}\n"
                        f"     Dense Score      : {dense:.3f}\n"
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
        results, rejected = [], []
        for h in dense_hits:
            dense = float(h.score)
            final_score = round(dense, 3)
            note, accepted = "-", False
            if dense >= 0.85:
                accepted, note = True, "Data yang Relevan Ditemukan"

            item = {
                "request_id": h.payload.get("request_id"),
                "organization_id": h.payload.get("organization_id"),
                "request_name": h.payload.get("request_name"),
                "request_rag_name": h.payload.get("request_rag_name"),
                "dense_score": dense,
                "final_score": final_score,
                "note": note
            }
            (results if accepted else rejected).append(item)

        results = sorted(results, key=lambda x: x["final_score"], reverse=True)
        rejected = sorted(rejected, key=lambda x: x["final_score"], reverse=True)

        # ==================================================
        # 🤖 AI TOPIC RELEVANCE CHECK
        # ==================================================
        if dense_hits:
            top_rag_q = dense_hits[0].payload.get("request_rag_name", "-")
            topic_check = ai_relevance_usulan(user_q, top_rag_q)
        else:
            topic_check = {"relevant": True, "reason": "Tidak ada hasil RAG"}

        if not topic_check.get("relevant", True):
            total_time = time.time() - t0
            logger.info(f"[AI-TOPIC-USULAN] Topik tidak relevan | Reason: {topic_check.get('reason')}")

            # ⭐ TAMBAHAN — LOG OUTPUT YANG AKAN DIKIRIM KE WABOT (kosong)
            logger.info("\n" + "=" * 60)
            logger.info("[USULAN-POST] Output akan dikirim ke WABOT: '-' (Tidak relevan)")
            logger.info("=" * 60)

            # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
            logger.info(f"[REQUEST] Total waktu permintaan: {total_time:.3f} detik")
            logger.info("=" * 60 + "\n")

            rag_summary_logger.info(
                f"\n{'='*60}\n[USULAN TOPIC CHECK]\nUser: {user_q}\nTopik RAG: {top_rag_q}\n"
                f"Relevan: {topic_check.get('relevant')} | Reason: {topic_check.get('reason')}\n{'='*60}\n"
            )
            return jsonify({
                "status": "low_confidence",
                "message": "Topik tidak relevan dengan pertanyaan pengguna",
                "reason": topic_check.get("reason", "-"),
                "data": {"similar_questions": []},
                "timing": {"total_sec": round(total_time, 3)}
            }), 200

        # ==================================================
        # ⭐ TAMBAHAN — LOG OUTPUT AKHIR YANG DIKIRIM KE WABOT
        # ==================================================
        if results:
            final_usulan = results[0]["request_rag_name"]
        else:
            final_usulan = "-"

        logger.info("\n" + "=" * 60)
        logger.info(f"[USULAN-POST] Output akan dikirim ke WABOT: '{final_usulan}'")
        logger.info("=" * 60)

        # ==================================================
        # ⏱️ WAKTU TOTAL & PAYLOAD
        # ==================================================
        total_time = time.time() - t0
        payload = {
            "status": "success" if results else "low_confidence",
            "message": "Hasil ditemukan" if results else "Tidak ada hasil cukup relevan",
            "data": {
                "similar_questions": results if results else rejected,
                "metadata": {
                    "wa_number": wa,
                    "user_question": user_q,
                    "final_score_top": (results[0]["final_score"] if results else "-")
                }
            },
            "timing": {
                "reform_sec": round(t_ref_time, 3),
                "embedding_sec": round(emb_time, 3),
                "qdrant_sec": round(qd_time, 3),
                "total_sec": round(total_time, 3)
            }
        }

        # ⭐ TAMBAHAN — LOG TOTAL WAKTU REQUEST
        logger.info(f"[REQUEST] Total waktu permintaan: {total_time:.3f} detik")
        logger.info("=" * 60 + "\n")

        # ==================================================
        # 🧾 LOG RINGKASAN
        # ==================================================
        try:
            summary_lines = [
                "\n" + "=" * 60,
                "[USULAN SEARCH SESSION]",
                f"Pertanyaan User: {user_q}",
                f"Hasil Pencarian: {len(dense_hits)} kandidat ditemukan",
                f"Hasil Cek Topik: {topic_check.get('relevant')} | Reason: {topic_check.get('reason')}",
            ]

            for idx, r in enumerate(results[:3], start=1):
                summary_lines.append(
                    f"{idx}. {r['request_rag_name']} | ReqID={r['request_id']} | Dense={r['dense_score']:.3f}"
                )

            req_ids = [r.get("request_id") for r in results[:3] if r.get("request_id")]
            summary_lines.append(f"Request ID yang dikembalikan: {req_ids}")
            summary_lines.append(
                f"Total waktu proses: {total_time:.3f} detik "
                f"(Reform={t_ref_time:.3f}s | Emb={emb_time:.3f}s | Qdrant={qd_time:.3f}s)"
            )
            summary_lines.append("=" * 60 + "\n")

            rag_summary_logger.info("\n".join(summary_lines))
        except Exception as e:
            logger.warning(f"[LOGGING ERROR] Gagal mencetak ringkasan USULAN: {e}")

        return jsonify(payload), 200

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][search_usulan] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal saat pencarian usulan", detail=str(e))
