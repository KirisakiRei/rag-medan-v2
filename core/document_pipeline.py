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
    local_file_path = _resolve_file(file_url)
    document_filename = Path(local_file_path).name

    logger.info("=" * 80)
    logger.info(f"[DOC] 🚀 Mulai proses dokumen | doc_id={doc_id} | opd={opd}")
    logger.info(f"[DOC] File sumber: {local_file_path}")


    # =====================================================
    # 1️⃣ OCR Multi-page Aman
    # =====================================================
    ocr_start = time.time()
    extracted_pages = extract_text_from_file(local_file_path, lang=lang, return_pages=True)
    ocr_duration = time.time() - ocr_start

    logger.info(f"[DOC] ✅ OCR selesai | {len(extracted_pages)} halaman | waktu {ocr_duration:.2f}s")


    # =====================================================
    # 2️⃣ Chunk per Halaman
    # =====================================================
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    text_chunks = []
    logger.info("[DOC] ✂️ Mulai chunking halaman...")

    for page_number, page_text in tqdm(extracted_pages.items(), desc="Chunking pages"):
        if not page_text.strip():
            continue
        split_text_chunks = text_splitter.split_text(page_text)

        for chunk_index, chunk_text in enumerate(split_text_chunks):
            text_chunks.append({
                "page_number": page_number,
                "chunk_index": chunk_index,
                "text": chunk_text.strip()
            })

    logger.info(f"[DOC] ✅ Chunking selesai | total chunks: {len(text_chunks)}")


    # =====================================================
    # 3️⃣ Smart Merge — Gabungkan chunk kecil
    # =====================================================
    merged_text_chunks = []
    text_buffer = ""
    buffer_page_number = None

    for chunk_item in text_chunks:
        chunk_text = chunk_item["text"]

        # buffer kosong → mulai buffer baru
        if not text_buffer:
            text_buffer = chunk_text
            buffer_page_number = chunk_item["page_number"]
            continue

        # halaman masih berdekatan + panjang OK → merge
        if abs(chunk_item["page_number"] - buffer_page_number) <= 1 and len(text_buffer) + len(chunk_text) < 1800:
            text_buffer += " " + chunk_text
        else:
            merged_text_chunks.append(text_buffer.strip())
            text_buffer = chunk_text
            buffer_page_number = chunk_item["page_number"]

    if text_buffer:
        merged_text_chunks.append(text_buffer.strip())

    logger.info(f"[DOC] 🔧 Merge selesai | merged chunks: {len(merged_text_chunks)}")


    # =====================================================
    # 4️⃣ Embedding + Build Qdrant Points
    # =====================================================
    logger.info("[DOC] 🧠 Embedding chunks...")

    qdrant_points = []
    for chunk_index, merged_chunk_text in enumerate(tqdm(merged_text_chunks, desc="Embedding")):

        embedding_vector = model.encode(f"passage: {merged_chunk_text}", normalize_embeddings=True).tolist()

        chunk_payload = {
            "mysql_id": doc_id,
            "opd": opd,
            "filename": document_filename,
            "page_number": chunk_index + 1,
            "chunk_index": chunk_index,
            "text": merged_chunk_text,
            "source_type": "document",
            "created_at": datetime.utcnow().isoformat()
        }

        qdrant_points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding_vector,
            payload=chunk_payload
        ))


    # =====================================================
    # 5️⃣ Upload ke Qdrant
    # =====================================================
    qdrant.upsert(
        collection_name=collection_name,
        wait=True,
        points=qdrant_points
    )

    total_duration = time.time() - start_time

    logger.info(f"[DOC] ✅ Upload selesai | {len(qdrant_points)} chunks → '{collection_name}'")
    logger.info(f"[PERF] Total: {total_duration:.2f}s | OCR={ocr_duration:.2f}s | Embedding={total_duration - ocr_duration:.2f}s")
    logger.info("=" * 80)

    return {
        "status": "ok",
        "filename": document_filename,
        "total_chunks": len(qdrant_points),
        "duration_sec": round(total_duration, 2)
    }


# ========================================================
# 🔹 Download atau baca file lokal
# ========================================================
def _resolve_file(url: str) -> str:
    """Detect local file or remote URL."""
    if url.startswith("file://"):
        local_path = url.replace("file://", "")
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file tidak ditemukan: {local_path}")
        return local_path

    if os.path.exists(url):
        return url

    # Download dari URL
    logger.info(f"[DOC] 🔽 Download file: {url}")
    download_start = time.time()

    http_response = requests.get(url, timeout=150)
    http_response.raise_for_status()

    file_suffix = Path(url).suffix or ".bin"
    file_descriptor, temp_file_path = tempfile.mkstemp(suffix=file_suffix)

    with os.fdopen(file_descriptor, "wb") as temp_file:
        temp_file.write(http_response.content)

    logger.info(f"[DOC] ✅ Download selesai ({len(http_response.content)/1024/1024:.2f} MB) dalam {time.time()-download_start:.2f}s")

    return temp_file_path
