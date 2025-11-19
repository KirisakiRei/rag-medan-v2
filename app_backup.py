from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.http import models
import logging, time, sys, os, traceback, requests
from config import CONFIG

from core.utils import (
    detect_category,
    normalize_text,
    clean_location_terms,
    keyword_overlap,
    safe_parse_answer_id,
)
from core.filtering import (
    ai_pre_filter, 
    ai_check_relevance, 
    ai_pre_filter_usulan, 
    ai_relevance_usulan
)

# Import blueprints
from routes.health_routes import health_bp
from routes.search_routes import search_bp
from routes.sync_routes import sync_bp
from routes.usulan_routes import usulan_bp

app = Flask(__name__)

@app.route("/")
def home():
    return "Server Flask App is running!"

# Initialize model dan qdrant sebagai global variables
# untuk bisa diakses oleh blueprints
model = SentenceTransformer(CONFIG["embeddings"]["model_path"])
qdrant = QdrantClient(
    host=CONFIG["qdrant"]["host"],
    port=CONFIG["qdrant"]["port"]
)

LOG_FILE = "/var/log/rag-medan.log"
SUMMARY_FILE = "/var/log/rag-summary.log"


MASTER_PID = os.getppid()  
CURRENT_PID = os.getpid()


if not logging.getLogger("app").handlers:
    logger = logging.getLogger("app")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    console_handler = logging.StreamHandler(sys.stdout)
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")

   
    formatter = logging.Formatter(
        f"%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    rag_summary_logger = logging.getLogger("rag.summary")
    rag_summary_logger.setLevel(logging.INFO)
    rag_summary_logger.propagate = False

    summary_handler = logging.FileHandler(SUMMARY_FILE, encoding="utf-8")
    summary_handler.setFormatter(logging.Formatter("%(asctime)s [INFO]: %(message)s"))
    rag_summary_logger.addHandler(summary_handler)

   
    for noisy in ["werkzeug", "httpx", "httpcore", "qdrant_client", "urllib3"]:
        lib_logger = logging.getLogger(noisy)
        lib_logger.handlers.clear()
        lib_logger.setLevel(logging.WARNING)
        lib_logger.propagate = True

    logger.info(
        f" Logging initialized (PID={CURRENT_PID}, Master={MASTER_PID}) — app log to rag-medan.log, summary log to rag-summary.log"
    )

else:

    logger = logging.getLogger("app")
    rag_summary_logger = logging.getLogger("rag.summary")


# ============================================================
# 🔹 Register Blueprints
# ============================================================
app.register_blueprint(health_bp)
app.register_blueprint(search_bp)
app.register_blueprint(sync_bp)
app.register_blueprint(usulan_bp)

logger.info("✅ All blueprints registered successfully")


# ============================================================
# 🔹 Legacy Routes (akan dihapus setelah verifikasi blueprints)
# ============================================================
@app.route("/health-legacy", methods=["GET"])
def health_check_legacy():
    try:
        _ = model.encode("health check").tolist()
        model_status = True
    except Exception as e:
        logger.error(f"[HEALTH] Model error: {e}", exc_info=True)
        model_status = False

    try:
        qdrant.get_collections()
        qdrant_status = True
    except Exception as e:
        logger.error(f"[HEALTH] Qdrant error: {e}", exc_info=True)
        qdrant_status = False

    overall_status = model_status and qdrant_status
    return jsonify({
        "status": "healthy" if overall_status else "unhealthy",
        "components": {
            "flask": True,
            "embedding_model": model_status,
            "qdrant": qdrant_status
        }
    }), 200 if overall_status else 500


# ============================================================
# ⚠️ NOTE: Semua endpoints sudah dipindahkan ke routes/
#    - /health → routes/health_routes.py
#    - /api/search → routes/search_routes.py
#    - /api/sync → routes/sync_routes.py
#    - /api/sync-usulan & /api/search-usulan → routes/usulan_routes.py
# ============================================================


if __name__ == "__main__":
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
        # ==================================================
        try:
            should_fallback = (
                payload["status"] == "low_confidence" or
                not payload["data"]["similar_questions"] or
                (results and results[0]["final_score"] < 0.85)
            )

            if should_fallback:
                logger.info("[FALLBACK] Tidak ada hasil cukup relevan di RAG teks → mencoba ke RAG dokumen")
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
                            # Format payload agar konsisten dengan RAG text
                            payload = {
                                "status": "success",
                                "message": "Hasil ditemukan dari dokumen",
                                "data": {
                                    "similar_questions": [{
                                        "question": f"[Dokumen] {top.get('filename', 'Unknown')} - Halaman {top.get('page_number', '-')}",
                                        "question_rag_name": top.get("text", "-"),
                                        "answer_id": None,
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
                                        "ai_reason": "Fallback ke RAG dokumen",
                                        "ai_reformulated": "-",
                                        "final_score_top": round(top.get("score", 0.0), 3)
                                    }
                                },
                                "timing": payload["timing"]
                            }
                            logger.info("[FALLBACK] ✅ Jawaban ditemukan di RAG dokumen")
                            rag_summary_logger.info(
                                f"[RAG FALLBACK] {user_q} → Dokumen: {top.get('filename')} (score={top.get('score'):.3f})"
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



@app.route("/api/sync", methods=["POST"])
def sync_data():
    try:
        data = request.json
        if not data or "action" not in data:
            return error_response("ValidationError", "Field 'action' wajib diisi", code=400)
        action = data["action"]
        content = data.get("content")

        if action == "bulk_sync":
            if not isinstance(content, list):
                return error_response("ValidationError", "Content harus berupa list", code=400)
            points = []
            for item in content:
                vector = model.encode("passage: " + item["question_rag_name"]).tolist()
                point_id = str(item["question_rag_id"])
                points.append({
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": item["question_id"],
                        "answer_id": item["answer_id"],
                        "category_id": item["category_id"],
                        "question": item["question"],
                        "question_rag_name": item["question_rag_name"]
                    }
                })
            qdrant.upsert(collection_name="knowledge_bank", points=points)
            qdrant.create_payload_index(
                collection_name="knowledge_bank",
                field_name="question_rag_name",
                field_schema=models.TextIndexParams(
                    type="text",
                    tokenizer=models.TokenizerType.WORD,
                    min_token_len=2,
                    max_token_len=15,
                    lowercase=True
                ))
            logger.info(f"[SYNC-DATA] Sinkronisasi {len(points)} data ke Knowledge Bank berhasil")
            return jsonify({
                "status": "success",
                "message": f"Sinkronisasi {len(points)} data berhasil",
                "total_synced": len(points)
            })
        elif action == "add":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil ditambahkan ke Knowledge Bank: ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil ditambahkan", "id": point_id})
        
        elif action == "update":
            point_id = str(content["question_rag_id"])
            vector = model.encode("passage: " + content["question_rag_name"]).tolist()
            qdrant.upsert(
                collection_name="knowledge_bank",
                points=[{
                    "id": point_id,
                    "vector": vector,
                    "payload": {
                        "question_id": content["question_id"],
                        "answer_id": content["answer_id"],
                        "category_id": content.get("category_id"),
                        "question": content["question"],
                        "question_rag_name": content["question_rag_name"]
                    }
                }]
            )
            logger.info(f"[SYNC-DATA] Data berhasil Diperbarui di Knowledge Bank: ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil diperbarui"})
        
        elif action == "delete":
            point_id = str(content["question_rag_id"])
            qdrant.delete(
                collection_name="knowledge_bank",
                points_selector=models.PointIdsList(points=[point_id]),
                wait=True
            )
            logger.info(f"[SYNC-DATA] Data dihapus : ID={point_id}")
            return jsonify({"status": "success", "message": "Data berhasil dihapus"})
        else:
            return error_response("ValidationError", f"Action '{action}' tidak dikenali", code=400)

    except Exception as e:
        err_trace = traceback.format_exc()
        logger.error(f"[ERROR][sync_data] {str(e)}\n{err_trace}")
        return error_response("ServerError", "Kesalahan internal saat sinkronisasi", detail=str(e))


# API SYNC-USULAN
@app.route("/api/sync-usulan", methods=["POST"])
def sync_usulan():
    try:
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

#  API SEARCH-USULAN
@app.route("/api/search-usulan", methods=["POST"])
def search_usulan():
    try:
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


if __name__ == "__main__":
    app.run(
        host=CONFIG["api"]["host"],
        port=CONFIG["api"]["port"],
        debug=False,
        use_reloader=False
    )

