import os, time, tempfile, requests, logging, uuid
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .ocr_utils import extract_text_from_file
from .embedding_utils import embed_passage

logger = logging.getLogger("document_pipeline")

def process_document(doc_id, opd, file_url, qdrant, model, lang="id",
                     collection_name="document_bank",
                     chunk_size=1000, chunk_overlap=100):
    """
    OCR dokumen, potong menjadi chunks, embedding, dan simpan ke Qdrant.
    Kompatibel untuk:
    - file:///home/...   (lokal)
    - https://server/... (remote)
    """
    t0 = time.time()
    tmp_path = None
    logger.info("=" * 80)
    logger.info(f"[DOC] 🚀 Mulai proses dokumen | doc_id={doc_id} | opd={opd}")
    logger.info(f"[DOC] Sumber file: {file_url}")

    try:
        # 🔹 1. Download / load file
        t1 = time.time()
        tmp_path = _resolve_file(file_url)
        filename = Path(tmp_path).name
        size_mb = os.path.getsize(tmp_path) / 1024 / 1024
        logger.info(f"[DOC] ✅ File siap diproses: {filename} ({size_mb:.2f} MB)")
        logger.info(f"[TIME] Step 1 selesai dalam {time.time() - t1:.2f}s")

        # 🔹 2. OCR
        logger.info(f"[DOC] 🔍 Mulai OCR file {filename} (bahasa={lang}) ...")
        t2 = time.time()
        text = extract_text_from_file(tmp_path, lang=lang)
        ocr_dur = time.time() - t2
        logger.info(f"[DOC] ✅ OCR selesai | durasi {ocr_dur:.2f}s | panjang teks {len(text):,} karakter")
        if not text or not text.strip():
            raise ValueError("OCR gagal atau teks kosong")

        # 🔹 3. Split teks
        logger.info(f"[DOC] ✂️  Mulai splitting teks (chunk_size={chunk_size}, overlap={chunk_overlap}) ...")
        t3 = time.time()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "."]
        )
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        split_dur = time.time() - t3
        logger.info(f"[DOC] ✅ Split selesai | total {len(chunks)} chunks | waktu {split_dur:.2f}s")

        # 🔹 4. Embedding + simpan ke Qdrant
        logger.info(f"[DOC] 🧠 Mulai proses embedding dan simpan ke Qdrant ...")
        t4 = time.time()
        points = []
        for i, ch in enumerate(tqdm(chunks, desc="🔹 Embedding Chunks", ncols=80)):
            vec = embed_passage(model, ch)
            points.append({
                "id": str(uuid.uuid4()),  # id unik tiap chunk
                "vector": vec,
                "payload": {
                    "mysql_id": doc_id,
                    "opd": opd,
                    "filename": filename,
                    "page_number": (i // 10) + 1,
                    "chunk_index": i,
                    "text": ch,
                    "source_type": "document",
                    "created_at": datetime.utcnow().isoformat()
                }
            })
        emb_dur = time.time() - t4
        logger.info(f"[DOC] ✅ Embedding selesai ({len(points)} points) | waktu {emb_dur:.2f}s")

        # Pastikan collection
        logger.info(f"[DOC] ⚙️  Verifikasi collection '{collection_name}' di Qdrant ...")
        size = len(points[0]["vector"])
        _ensure_collection(qdrant, collection_name, size)
        logger.info(f"[DOC] ✅ Collection OK (vector dim={size})")

        logger.info(f"[DOC] 📤 Upload data ke Qdrant ...")
        t5 = time.time()
        qdrant.upsert(collection_name=collection_name, points=points)
        upsert_dur = time.time() - t5
        logger.info(f"[DOC] ✅ Upload selesai | waktu {upsert_dur:.2f}s")

        dur = round(time.time() - t0, 3)
        logger.info(f"[DOC] 🎉 Dokumen {filename} selesai diproses.")
        logger.info(f"[SUMMARY] Total waktu {dur:.2f}s | OCR={ocr_dur:.2f}s | Split={split_dur:.2f}s | Embed+Upload={emb_dur+upsert_dur:.2f}s")
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
            os.remove(tmp_path)
            logger.debug(f"[DOC] 🧹 File sementara dihapus: {tmp_path}")


def _resolve_file(url: str) -> str:
    """Deteksi otomatis file lokal (file://) atau URL remote."""
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


def _ensure_collection(qdrant, name: str, size: int):
    """Pastikan collection ada di Qdrant."""
    try:
        qdrant.get_collection(name)
    except Exception:
        logger.warning(f"[DOC] Collection '{name}' tidak ditemukan, membuat ulang ...")
        qdrant.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
        logger.info(f"[DOC] ✅ Collection '{name}' dibuat ulang (dim={size})")
