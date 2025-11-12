import os, time, logging, uuid
from tqdm import tqdm
from pathlib import Path
from paddleocr import PaddleOCR
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .ocr_utils import extract_text_from_file
from .summarizer_utils import summarize_text  # pakai summarizer kamu yg sudah ada

logger = logging.getLogger("document_pipeline")

# ========================================================
# 🔹 OCR Helper — baca teks per halaman
# ========================================================
def extract_text_per_page(file_path, lang="id"):
    """
    Jalankan OCR per halaman PDF.
    Return: dict {page_number: text}
    """
    ocr = PaddleOCR(lang=lang, use_angle_cls=True, show_log=False)  
    pages = ocr.ocr(file_path, cls=True)
    result = {}

    for i, page in enumerate(pages, start=1):
        text = ""
        for line in page:
            if line and len(line) > 0:
                text += line[1][0] + " "
        result[i] = text.strip()
    return result


# ========================================================
# 🔹 Fungsi utama: proses dokumen OCR → Chunk → Summarize → Embed → Upload
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
    Proses lengkap dokumen:
    - OCR per halaman
    - Chunk per halaman
    - Merge chunk kecil
    - Summarization
    - Embedding
    - Upload ke Qdrant
    """
    start_time = time.time()
    tmp_path = file_url.replace("file://", "")

    logger.info("=" * 80)
    logger.info(f"[DOC] 🚀 Mulai proses dokumen | doc_id={doc_id} | opd={opd}")
    logger.info(f"[DOC] File sumber: {tmp_path}")

    # =====================================================
    # 1️⃣ OCR per halaman
    # =====================================================
    t0 = time.time()
    pages = extract_text_per_page(tmp_path, lang=lang)
    ocr_time = time.time() - t0
    logger.info(f"[DOC] ✅ OCR selesai ({len(pages)} halaman) | waktu {ocr_time:.2f}s")

    # =====================================================
    # 2️⃣ Chunking per halaman
    # =====================================================
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " "]
    )

    chunks = []
    logger.info("[DOC] ✂️ Mulai chunking per halaman ...")

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

    logger.info(f"[DOC] ✅ Chunking selesai | total {len(chunks)} chunks | waktu {time.time() - t0:.2f}s")

    # =====================================================
    # 3️⃣ Smart Merge (gabung chunk kecil yang masih satu konteks)
    # =====================================================
    merged_chunks = []
    buffer = ""
    buffer_page = None

    for c in chunks:
        text = c["text"]
        # kalau buffer kosong → isi
        if not buffer:
            buffer = text
            buffer_page = c["page_number"]
            continue

        # kalau halaman berurutan dan gabungan < 1800 karakter → merge
        if abs(c["page_number"] - buffer_page) <= 1 and len(buffer) + len(text) < 1800:
            buffer += " " + text
        else:
            merged_chunks.append(buffer.strip())
            buffer = text
            buffer_page = c["page_number"]

    # jangan lupa masukkan buffer terakhir
    if buffer:
        merged_chunks.append(buffer.strip())

    logger.info(f"[DOC] 🔧 Merge selesai | total {len(merged_chunks)} merged chunks")

    # =====================================================
    # 4️⃣ Summarization per chunk
    # =====================================================
    logger.info("[DOC] 📝 Mulai summarization per chunk ...")
    summarized = []
    sum_start = time.time()

    for i, chunk_text in enumerate(tqdm(merged_chunks, desc="Summarizing")):
        try:
            summary = summarize_text(chunk_text)
        except Exception as e:
            logger.warning(f"[SUM] Gagal summarize chunk {i}: {e}")
            summary = chunk_text[:400] + "..."
        summarized.append({
            "text": chunk_text,
            "summary": summary
        })

    sum_time = time.time() - sum_start
    logger.info(f"[DOC] ✅ Summarization selesai | waktu {sum_time:.2f}s")

    # =====================================================
    # 5️⃣ Embedding & siapkan points untuk Qdrant
    # =====================================================
    logger.info("[DOC] 🧠 Mulai embedding & siapkan points untuk Qdrant ...")
    embed_start = time.time()
    points = []

    for i, item in enumerate(tqdm(summarized, desc="Embedding")):
        vec = embed_passage(model, item["text"])
        payload = {
            "mysql_id": doc_id,
            "opd": opd,
            "filename": os.path.basename(tmp_path),
            "page_number": i + 1,
            "chunk_index": i,
            "section": None,
            "summary": item["summary"],
            "text": item["text"],
        }
        points.append(models.PointStruct(
            id=str(uuid.uuid4()),
            vector=vec,
            payload=payload
        ))

    # =====================================================
    # 6️⃣ Upload ke Qdrant
    # =====================================================
    qdrant.upsert(
        collection_name=collection_name,
        wait=True,
        points=points
    )

    embed_time = time.time() - embed_start
    total_time = time.time() - start_time

    logger.info(f"[DOC] ✅ Upload selesai | {len(points)} points ke '{collection_name}'")
    logger.info(f"[SUMMARY] Total waktu {total_time:.2f}s | OCR={ocr_time:.2f}s | Chunk+Sum+Embed={total_time - ocr_time:.2f}s")
    logger.info("=" * 80)

    return {"status": "ok", "total_chunks": len(points)}


def embed_passage(model, text: str):
    """
    Buat embedding dari teks dokumen (OCR) menggunakan format standar E5.
    Format 'passage:' ini penting agar hasil embedding konsisten dengan query.
    """
    try:
        return model.encode(f"passage: {text}", normalize_embeddings=True).tolist()
    except Exception as e:
        logger.warning(f"[EMBED] Gagal membuat embedding: {e}")
        return []