import os, time, tempfile, requests, logging, uuid
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter

# OCR STABLE dari utils revisi
from .ocr_utils import extract_text_from_file

logger = logging.getLogger("document_pipeline")


# ========================================================
# 🔥 FINAL Pipeline: OCR → Chunk → Merge → Embed → Upload
# ========================================================
def process_document(
    doc_id,
    opd,
    file_url,
    qdrant,
    model,
    lang="id",
    collection_name="document_bank",
    chunk_size=1200,
    chunk_overlap=150
):
    """
    PRODUCTION SAFE PIPELINE:
    - OCR Multi-page (anti-crash, resize besar otomatis)
    - Chunking
    - Smart merge chunk pendek
    - Embedding
    - Upload Qdrant
    TANPA penggunaan LLM (super cepat + stabil)
    """

    start_time = time.time()
    tmp_path = _resolve_file(file_url)
    filename = Path(tmp_path).name

    logger.info("=" * 80)
    logger.info(f"[DOC] 🚀 Mulai proses dokumen | doc_id={doc_id} | opd={opd}")
    logger.info(f"[DOC] File sumber: {tmp_path}")


    # =====================================================
    # 1️⃣ OCR Multi-page Aman
    # =====================================================
    t0 = time.time()
    pages = extract_text_from_file(tmp_path, lang=lang, return_pages=True)
    ocr_time = time.time() - t0

    logger.info(f"[DOC] ✅ OCR selesai | {len(pages)} halaman | waktu {ocr_time:.2f}s")


    # =====================================================
    # 2️⃣ Chunk per Halaman
    # =====================================================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    logger.info("[DOC] ✂️ Mulai chunking halaman...")

    for page_num, text in tqdm(pages.items(), desc="Chunking pages"):
        if not text.strip():
            continue
        split_chunks = splitter.split_text(text)

        for i, ch in enumerate(split_chunks):
            chunks.append({
                "page_number": page_num,
                "chunk_index": i,
                "text": ch.strip()
            })

    logger.info(f"[DOC] ✅ Chunking selesai | total chunks: {len(chunks)}")


    # =====================================================
    # 3️⃣ Smart Merge — Gabungkan chunk kecil
    # =====================================================
    merged_chunks = []
    buffer = ""
    buffer_page = None

    for c in chunks:
        text = c["text"]

        # buffer kosong → mulai buffer baru
        if not buffer:
            buffer = text
            buffer_page = c["page_number"]
            continue

        # halaman masih berdekatan + panjang OK → merge
        if abs(c["page_number"] - buffer_page) <= 1 and len(buffer) + len(text) < 1800:
            buffer += " " + text
        else:
            merged_chunks.append(buffer.strip())
            buffer = text
            buffer_page = c["page_number"]

    if buffer:
        merged_chunks.append(buffer.strip())

    logger.info(f"[DOC] 🔧 Merge selesai | merged chunks: {len(merged_chunks)}")


    # =====================================================
    # 4️⃣ Embedding + Build Qdrant Points
    # =====================================================
    logger.info("[DOC] 🧠 Embedding chunks...")

    points = []
    for i, chunk_text in enumerate(tqdm(merged_chunks, desc="Embedding")):

        vec = model.encode(f"passage: {chunk_text}", normalize_embeddings=True).tolist()

        payload = {
            "mysql_id": doc_id,
            "opd": opd,
            "filename": filename,
            "page_number": i + 1,
            "chunk_index": i,
            "text": chunk_text,
            "source_type": "document",
            "created_at": datetime.utcnow().isoformat()
        }

        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload=payload
        ))


    # =====================================================
    # 5️⃣ Upload ke Qdrant
    # =====================================================
    qdrant.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    total_time = time.time() - start_time

    logger.info(f"[DOC] ✅ Upload selesai | {len(points)} chunks → '{collection_name}'")
    logger.info(f"[PERF] Total: {total_time:.2f}s | OCR={ocr_time:.2f}s | Embedding={total_time - ocr_time:.2f}s")
    logger.info("=" * 80)

    return {
        "status": "ok",
        "filename": filename,
        "total_chunks": len(points),
        "duration_sec": round(total_time, 2)
    }


# ========================================================
# 🔹 Download atau baca file lokal
# ========================================================
def _resolve_file(url: str) -> str:
    """Detect local file or remote URL."""
    if url.startswith("file://"):
        path = url.replace("file://", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file tidak ditemukan: {path}")
        return path

    if os.path.exists(url):
        return url

    # Download dari URL
    logger.info(f"[DOC] 🔽 Download file: {url}")
    t = time.time()

    resp = requests.get(url, timeout=150)
    resp.raise_for_status()

    suffix = Path(url).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)

    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)

    logger.info(f"[DOC] ✅ Download selesai ({len(resp.content)/1024/1024:.2f} MB) dalam {time.time()-t:.2f}s")

    return tmp
