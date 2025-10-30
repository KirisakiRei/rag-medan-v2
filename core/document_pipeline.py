import os, time, tempfile, requests
from pathlib import Path
from datetime import datetime
from qdrant_client.http import models
from langchain.text_splitter import RecursiveCharacterTextSplitter
from .ocr_utils import extract_text_from_file
from .embedding_utils import embed_passage


def process_document(
    doc_id,
    opd,
    file_url,
    qdrant,
    model,
    lang="id",
    collection_name="document_bank",
    chunk_size=1000,
    chunk_overlap=100,
):
    """
    OCR dokumen, potong menjadi chunks per halaman, embedding, dan simpan ke Qdrant.
    """
    t0 = time.time()
    tmp_path = None
    try:
        # 🔹 Download file temporer
        tmp_path = _download_temp(file_url)
        filename = Path(tmp_path).name

        # 🔹 OCR per halaman (dict: {page_number: text})
        texts = extract_text_from_file(tmp_path, lang=lang)
        if not texts or not any(v.strip() for v in texts.values()):
            return {"status": "error", "message": "OCR gagal atau teks kosong"}

        points = []
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "."],
        )

        # 🔹 Loop setiap halaman dan bagi teks jadi chunk
        for page_number, page_text in texts.items():
            chunks = [c for c in splitter.split_text(page_text) if c.strip()]
            for i, ch in enumerate(chunks):
                vec = embed_passage(model, ch)
                points.append({
                    "id": f"{doc_id}-{page_number}-{i}",
                    "vector": vec,
                    "payload": {
                        "doc_id": doc_id,                         # UUID MySQL
                        "opd": opd,                               # Nama OPD
                        "filename": filename,                     # Nama file sumber
                        "page_number": page_number,               # ✅ halaman real
                        "chunk_index": i,                         # urutan chunk
                        "text": ch,                               # isi teks OCR
                        "created_at": datetime.utcnow().isoformat()
                    },
                })

        if not points:
            return {"status": "error", "message": "Tidak ada potongan teks valid"}

        # 🔹 Pastikan collection sudah ada
        size = len(points[0]["vector"])
        _ensure_collection(qdrant, collection_name, size)

        # 🔹 Simpan ke Qdrant
        qdrant.upsert(collection_name=collection_name, points=points)

        return {
            "status": "ok",
            "total_pages": len(texts),
            "total_chunks": len(points),
            "filename": filename,
            "duration_sec": round(time.time() - t0, 3),
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _download_temp(url: str) -> str:
    """
    Download file dari URL ke temporary directory,
    atau baca langsung dari local path jika bukan HTTP(S).
    """
    # 🔹 Deteksi apakah URL lokal
    if url.startswith("file://"):
        path = url.replace("file://", "")
    elif url.startswith("/"):
        path = url
    else:
        # 🔹 Kalau dari internet
        resp = requests.get(url, timeout=120)
        resp.raise_for_status()
        suffix = Path(url).suffix or ".bin"
        fd, tmp = tempfile.mkstemp(suffix=suffix)
        with os.fdopen(fd, "wb") as f:
            f.write(resp.content)
        return tmp

    # 🔹 Kalau lokal file path
    if not os.path.exists(path):
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    suffix = Path(path).suffix
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with open(path, "rb") as src, os.fdopen(fd, "wb") as dst:
        dst.write(src.read())
    return tmp



def _ensure_collection(qdrant, name: str, size: int):
    """Pastikan collection Qdrant sudah ada, kalau belum buat ulang."""
    try:
        qdrant.get_collection(name)
    except Exception:
        qdrant.recreate_collection(
            collection_name=name,
            vectors_config=models.VectorParams(
                size=size,
                distance=models.Distance.COSINE
            ),
        )
