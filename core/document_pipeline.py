import os, time, tempfile, requests, logging
from pathlib import Path
from datetime import datetime
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
    logger.info(f"[DOC] Mulai proses dokumen doc_id={doc_id}, opd={opd}, url={file_url}")

    try:
        # 🔹 1. Download / load file
        tmp_path = _resolve_file(file_url)
        filename = Path(tmp_path).name
        logger.info(f"[DOC] File siap diproses: {filename}")

        # 🔹 2. OCR
        text = extract_text_from_file(tmp_path, lang=lang)
        if not text or not text.strip():
            raise ValueError("OCR gagal atau teks kosong")

        # 🔹 3. Split teks
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "."]
        )
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        logger.info(f"[DOC] Total chunk hasil split: {len(chunks)}")

        # 🔹 4. Embedding + simpan ke Qdrant
        points = []
        for i, ch in enumerate(chunks):
            vec = embed_passage(model, ch)
            points.append({
                "id": f"{doc_id}-{i}",
                "vector": vec,
                "payload": {
                    "doc_id": doc_id,
                    "opd": opd,
                    "filename": filename,
                    "page_number": (i // 10) + 1,
                    "chunk_index": i,
                    "text": ch,
                    "created_at": datetime.utcnow().isoformat()
                }
            })

        # Pastikan collection
        size = len(points[0]["vector"])
        _ensure_collection(qdrant, collection_name, size)

        qdrant.upsert(collection_name=collection_name, points=points)
        dur = round(time.time() - t0, 3)
        logger.info(f"[DOC] ✅ Dokumen {filename} selesai | {len(points)} chunks | {dur}s")

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


def _resolve_file(url: str) -> str:
    """Deteksi otomatis file lokal (file://) atau URL remote."""
    if url.startswith("file://"):
        path = url.replace("file://", "")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Local file not found: {path}")
        return path

    if os.path.exists(url):
        return url

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    suffix = Path(url).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
    return tmp


def _ensure_collection(qdrant, name: str, size: int):
    """Pastikan collection ada di Qdrant."""
    try:
        qdrant.get_collection(name)
    except Exception:
        qdrant.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(size=size, distance=models.Distance.COSINE)
        )
