# ============================================================
# ✅ Document Pipeline — v2 (upgrade internal, tanpa ubah API)
# ============================================================
import os, time, tempfile, requests, logging, uuid, math
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import util
from config import CONFIG
from .ocr_utils import extract_text_from_file

logger = logging.getLogger("document_pipeline")

# ============================================================
# 🔹 Helper: Embed Passage (tetap sama)
# ============================================================
def embed_passage(model, text: str):
    """
    Buat embedding dari teks dokumen (OCR) menggunakan format standar E5.
    """
    return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()

# ============================================================
# 🔹 Helper: Summarization ringan via LLM (fallback aman)
# ============================================================
def _summarize_text(text: str, max_sentences: int = 3) -> str:
    """
    Ringkas teks dengan memanggil LLM dari CONFIG['llm'].
    Fallback: ambil 3 kalimat pertama jika LLM gagal.
    """
    try:
        base_url = CONFIG["llm"]["base_url"]
        api_key  = CONFIG["llm"]["api_key"]
        model    = CONFIG["llm"]["model"]
        timeout  = CONFIG["llm"].get("timeout_sec", 60)

        system_prompt = (
            "Anda adalah peringkas dokumen pemerintahan. "
            "Ringkas isi berikut secara formal, padat, dan jelas (maks 3 kalimat). "
            "Hindari opini; fokus pada inti informasi."
        )
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text[:4000]}  # batasi agar aman token
            ],
            "temperature": 0.0,
            "top_p": 0.9
        }
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(base_url, json=payload, headers=headers, timeout=timeout)
        if resp.status_code == 200:
            content = (resp.json().get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            if content and content.strip():
                # rapikan singkat
                return " ".join(content.strip().split())
    except Exception as e:
        logger.warning(f"[SUMMARIZE] fallback active: {e}")

    # fallback: ambil beberapa kalimat awal
    import re
    sents = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(sents[:max_sentences]).strip()

# ============================================================
# 🔹 Helper: Deteksi section (BAB/PASAL) sederhana
# ============================================================
def _detect_section(text: str) -> str:
    import re
    # cari pola umum: BAB, PASAL, BAGIAN, SUBBAGIAN, dll.
    m = re.search(r"(BAB\s+[IVXLC\d]+[^\n]*|PASAL\s+\d+|BAGIAN\s+[A-Z0-9]+[^\n]*)", text, flags=re.IGNORECASE)
    return m.group(0).strip() if m else ""

# ============================================================
# 🔹 Helper: Semantic merge antar halaman
# ============================================================
def _semantic_merge_pages(model, pages: Dict[int, str], sim_threshold: float = 0.80,
                          base_chunk_size: int = 1200, base_overlap: int = 150) -> List[Dict[str, Any]]:
    """
    Kembalikan list chunk:
    [
      {"text": "...", "page_start": 3, "page_end": 5, "section": "BAB II ..."},
      ...
    ]
    Step:
    - Split per halaman menjadi potongan kecil (Recursive splitter)
    - Lalu gabungkan potongan berurutan bila semantik serupa (cosine sim >= threshold)
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=base_chunk_size,
        chunk_overlap=base_overlap,
        separators=["\n\n", "\n", ". ", "."]
    )

    chunks: List[Dict[str, Any]] = []
    buffer_text, buffer_start, buffer_end, buffer_section = "", None, None, ""

    # siapkan list (page_num, part_text)
    items: List[Tuple[int, str]] = []
    for pnum in sorted(pages.keys()):
        page_text = pages[pnum].strip()
        if not page_text:
            continue
        parts = [p for p in splitter.split_text(page_text) if p.strip()]
        for idx, part in enumerate(parts):
            items.append((pnum, part))

    if not items:
        return chunks

    # inisialisasi buffer
    buffer_text = items[0][1]
    buffer_start = buffer_end = items[0][0]
    buffer_section = _detect_section(buffer_text)

    def _flush():
        nonlocal buffer_text, buffer_start, buffer_end, buffer_section
        if buffer_text.strip():
            chunks.append({
                "text": buffer_text.strip(),
                "page_start": int(buffer_start),
                "page_end": int(buffer_end),
                "section": buffer_section or _detect_section(buffer_text) or ""
            })
        buffer_text, buffer_start, buffer_end, buffer_section = "", None, None, ""

    # merge maju
    # encode pertama di luar loop untuk hemat
    prev_vec = None
    try:
        prev_vec = embed_passage(model, buffer_text)
    except Exception:
        prev_vec = None

    for i in range(1, len(items)):
        pnum, part = items[i]
        # similarity antara buffer dan part berikutnya
        try:
            curr_vec = embed_passage(model, part)
            if prev_vec is not None and curr_vec is not None:
                # cosine similarity (sudah L2-normalized)
                sim = float(util.cos_sim([prev_vec], [curr_vec]).cpu().numpy()[0][0])
            else:
                sim = 0.0
        except Exception:
            sim = 0.0

        if sim >= sim_threshold:
            # gabungkan
            buffer_text = f"{buffer_text}\n{part}"
            buffer_end = pnum
            # update section jika kosong
            if not buffer_section:
                buffer_section = _detect_section(part) or buffer_section
            prev_vec = curr_vec
        else:
            # flush buffer → mulai baru
            _flush()
            buffer_text = part
            buffer_start = buffer_end = pnum
            buffer_section = _detect_section(part)
            prev_vec = embed_passage(model, buffer_text)

    # flush terakhir
    _flush()
    return chunks

# ============================================================
# 🔹 Proses Dokumen Lengkap (signature TETAP)
# ============================================================
def process_document(doc_id, opd, file_url, qdrant, model, lang="id",
                     collection_name="document_bank",
                     chunk_size=1000, chunk_overlap=100):
    """
    OCR dokumen, contextual chunking, summarization, embedding batch, dan simpan ke Qdrant.
    Signature & alur umum TETAP; internal ditingkatkan.
    """
    t0 = time.time()
    tmp_path = None
    logger.info("=" * 80)
    logger.info(f"[DOC] 🚀 Mulai proses dokumen | doc_id={doc_id} | opd={opd}")
    logger.info(f"[DOC] Sumber file: {file_url}")

    try:
        # 1) Download / Load File
        t1 = time.time()
        tmp_path = _resolve_file(file_url)
        filename = Path(tmp_path).name
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        logger.info(f"[DOC] ✅ File siap diproses: {filename} ({size_mb:.2f} MB)")
        logger.info(f"[TIME] Step 1 selesai dalam {time.time() - t1:.2f}s")

        # 2) OCR Ekstraksi Teks (per halaman)
        logger.info(f"[DOC] 🔍 Mulai OCR file {filename} (bahasa={lang}) ...")
        t2 = time.time()
        # gunakan return_pages=True untuk dapat total halaman; lalu potong per halaman di _semantic_merge_pages
        full_text, _ = extract_text_from_file(tmp_path, lang=lang, return_pages=True)
        # kita butuh pages dict → ambil ulang halaman via util internal di ocr_utils (sudah dipanggil di atas)
        # untuk menjaga kompatibilitas, panggil lagi dengan return_pages supaya jumlah diketahui,
        # lalu potong kembali dari full_text jika perlu.
        # Cara yang lebih tepat: buka lagi PDF dan re-ekstrak map halaman → untuk efisiensi, kita buat ulang ringan di sini.
        # Namun agar TETAP sederhana & konsisten, kita buat ulang map via PyMuPDF langsung.
        import fitz
        pages_map = {}
        pdf_doc = fitz.open(tmp_path)
        # ambil text native atau OCR ringan per halaman (cepat, tanpa gambar jika sudah ada text)
        for page_num, page in enumerate(pdf_doc, start=1):
            text = page.get_text("text").strip()
            pages_map[page_num] = text if text else ""  # jika kosong, tetap kosong (sudah diambil di pass pertama)
        # kalau banyak kosong (scan), fallback: pecah full_text kasar per halaman (agar tidak crash)
        if sum(1 for v in pages_map.values() if v) < max(1, int(0.3 * len(pages_map))):
            # fallback sederhana: bagi rata (approx) — tetap aman, merger semantik akan merapikan
            approx = full_text.split("\n\n")
            chunk_per_page = max(1, len(approx) // max(1, len(pages_map)))
            idx = 0
            for p in pages_map.keys():
                seg = approx[idx: idx + chunk_per_page]
                pages_map[p] = "\n\n".join(seg).strip()
                idx += chunk_per_page

        ocr_dur = time.time() - t2
        logger.info(f"[DOC] ✅ OCR selesai | durasi {ocr_dur:.2f}s | panjang teks {len(full_text):,} karakter")
        if not full_text or not full_text.strip():
            raise ValueError("OCR gagal atau teks kosong")

        # 3) Contextual Chunking (semantic merge antar halaman)
        logger.info("[DOC] ✂️  Mulai contextual chunking (semantic merge per halaman) ...")
        t3 = time.time()
        merged_chunks = _semantic_merge_pages(
            model=model,
            pages=pages_map,
            sim_threshold=0.80,
            base_chunk_size=chunk_size,
            base_overlap=chunk_overlap
        )
        split_dur = time.time() - t3
        logger.info(f"[DOC] ✅ Chunking selesai | total {len(merged_chunks)} chunks | waktu {split_dur:.2f}s")

        # 4) Summarization per chunk
        logger.info("[DOC] 📝 Mulai summarization per chunk ...")
        t4 = time.time()
        for ch in merged_chunks:
            ch["summary"] = _summarize_text(ch["text"])
        sum_dur = time.time() - t4
        logger.info(f"[DOC] ✅ Summarization selesai | waktu {sum_dur:.2f}s")

        # 5) Embedding (batch) + siapkan points
        logger.info("[DOC] 🧠 Mulai batch embedding & siapkan points untuk Qdrant ...")
        t5 = time.time()
        texts_for_embed = [c["text"] for c in merged_chunks]
        # batch encode (hemat waktu)
        vectors = model.encode([f"passage: {t}" for t in texts_for_embed],
                               normalize_embeddings=True,
                               batch_size=32)
        points = []
        for i, ch in enumerate(tqdm(merged_chunks, desc="🔹 Build Points", ncols=80)):
            vec = vectors[i].tolist()
            points.append({
                "id": str(uuid.uuid4()),
                "vector": vec,
                "payload": {
                    "mysql_id": doc_id,
                    "opd": opd,
                    "filename": filename,
                    "page_number": ch["page_start"],     # tetap kompatibel
                    "page_start": ch["page_start"],
                    "page_end": ch["page_end"],
                    "chunk_index": i,
                    "section": ch.get("section", ""),
                    "summary": ch.get("summary", ""),
                    "text": ch["text"],
                    "source_type": "document",
                    "created_at": datetime.utcnow().isoformat()
                }
            })
        emb_dur = time.time() - t5
        logger.info(f"[DOC] ✅ Embedding selesai ({len(points)} points) | waktu {emb_dur:.2f}s")

        # 6) Pastikan Collection di Qdrant
        logger.info(f"[DOC] ⚙️  Verifikasi collection '{collection_name}' di Qdrant ...")
        size = len(points[0]["vector"])
        _ensure_collection(qdrant, collection_name, size)
        logger.info(f"[DOC] ✅ Collection OK (vector dim={size})")

        # 7) Upload ke Qdrant (batch aman)
        logger.info("[DOC] 📤 Upload data ke Qdrant ...")
        t6 = time.time()
        # upsert langsung (Qdrant client sudah batch-friendly)
        qdrant.upsert(collection_name=collection_name, points=points)
        upsert_dur = time.time() - t6
        logger.info(f"[DOC] ✅ Upload selesai | waktu {upsert_dur:.2f}s")

        # 8) Ringkasan Proses
        dur = round(time.time() - t0, 3)
        logger.info(f"[DOC] 🎉 Dokumen {filename} selesai diproses.")
        logger.info(f"[SUMMARY] Total waktu {dur:.2f}s | "
                    f"OCR={ocr_dur:.2f}s | Chunk={split_dur:.2f}s | "
                    f"Sum={sum_dur:.2f}s | Embed+Upload={(emb_dur+upsert_dur):.2f}s")
        logger.info("=" * 80)

        return {
            "status": "ok",
            "filename": filename,
            "total_chunks": len(points),
            "duration_sec": dur
        }

    except Exception as e:
        logger.exception(f"[DOC] ❌ Error: {str(e)}")
        return {"status": "error", "message": str(e)}

    finally:
        if tmp_path and tmp_path.startswith("/tmp") and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
                logger.debug(f"[DOC] 🧹 File sementara dihapus: {tmp_path}")
            except Exception:
                pass

# ============================================================
# 🔹 Helper — Download / Resolusi File (tetap)
# ============================================================
def _resolve_file(url: str) -> str:
    if url.startswith("file://"):
        path = url.replace("file://", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file not found: {path}")
        return path

    if os.path.exists(url):
        return url

    logger.info(f"[DOC] 🔽 Mengunduh file dari URL: {url}")
    t = time.time()
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    suffix = Path(url).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    logger.info(f"[DOC] ✅ Unduh selesai ({len(resp.content)/1024/1024:.2f} MB) dalam {time.time()-t:.2f}s")
    return tmp

# ============================================================
# 🔹 Helper — Pastikan Collection di Qdrant (tetap)
# ============================================================
def _ensure_collection(qdrant, name: str, size: int):
    try:
        qdrant.get_collection(name)
    except Exception:
        logger.warning(f"[DOC] Collection '{name}' tidak ditemukan, membuat ulang ...")
        qdrant.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
        logger.info(f"[DOC] ✅ Collection '{name}' dibuat ulang (dim={size})")
