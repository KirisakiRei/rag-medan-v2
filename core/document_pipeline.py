# core/document_pipeline.py

import os
import time
import tempfile
import requests
import uuid
from pathlib import Path
from datetime import datetime
from qdrant_client.http import models
# gunakan text splitter sesuai versi langchain yang terpasang
from langchain.text_splitter import RecursiveCharacterTextSplitter

from .ocr_utils import extract_text_from_file
from .embedding_utils import embed_passage

# gunakan module logger (doc_app akan mengatur logging)
import logging
logger = logging.getLogger("doc_app.document_pipeline")


def process_document(doc_id, opd, file_url, qdrant, model, lang="id",
                     collection_name="document_bank", chunk_size=1000, chunk_overlap=100):
    """
    OCR dokumen -> split -> embed -> upsert ke Qdrant
    Mengembalikan dict dengan rincian progress dan timing.
    """
    t0 = time.time()
    tmp_path = None
    progress = {
        "doc_id": doc_id,
        "filename": None,
        "total_pages": None,
        "total_chunks": 0,
        "chunks": [],   # list of dicts per chunk: {index, page, tokens, duration_sec, status}
        "stages": {},   # timing per major stage
    }

    try:
        logger.info(f"[DOC] Mulai proses dokumen doc_id={doc_id} url={file_url}")
        # ---------- download temp ----------
        t_dl = time.time()
        tmp_path = _download_temp(file_url)
        progress["filename"] = Path(tmp_path).name
        progress["stages"]["download_sec"] = round(time.time() - t_dl, 3)
        logger.info(f"[DOC] File di-download -> {progress['filename']} ({progress['stages']['download_sec']}s)")

        # ---------- extract text (OCR / PDF text) ----------
        t_ocr_start = time.time()
        text, page_text_map = extract_text_from_file(tmp_path, lang=lang, return_pages=True)
        # page_text_map : dict mapping page_number -> text_of_page (if backend returns that)
        t_ocr = time.time() - t_ocr_start
        progress["stages"]["ocr_sec"] = round(t_ocr, 3)
        if page_text_map:
            progress["total_pages"] = len(page_text_map)
            logger.info(f"[DOC] OCR selesai, halaman terdeteksi: {progress['total_pages']}")
        else:
            logger.info(f"[DOC] OCR selesai (no per-page map), total text length: {len(text or '')}")

        if not text or not text.strip():
            logger.warning("[DOC] OCR gagal atau teks kosong")
            return {"status": "error", "message": "OCR gagal atau teks kosong", "progress": progress}

        # ---------- split text menjadi chunks ----------
        t_split_start = time.time()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "."]
        )
        chunks = [c for c in splitter.split_text(text) if c.strip()]
        t_split = time.time() - t_split_start
        progress["stages"]["split_sec"] = round(t_split, 3)
        logger.info(f"[DOC] Split text -> {len(chunks)} chunks (chunk_size={chunk_size}, overlap={chunk_overlap})")

        if not chunks:
            return {"status": "error", "message": "Tidak ada chunk yang valid", "progress": progress}

        # ---------- prepare and embed each chunk (log per chunk) ----------
        points = []
        for i, ch in enumerate(chunks):
            chunk_start = time.time()
            # embed passage (embed_passage harus mengembalikan list/np.array)
            vec = embed_passage(model, ch)
            chunk_dur = time.time() - chunk_start

            # deduce page_number if we have page_text_map (best-effort)
            page_number = None
            if page_text_map:
                # simple heuristic: find which page contain the chunk text snippet
                # we check first 100 chars to find match
                snippet = ch[:200].strip()
                for p, ptext in page_text_map.items():
                    if snippet and snippet in ptext:
                        page_number = p
                        break

            # create deterministic/stable uuid for qdrant point id based on doc_id + index
            qpoint_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{doc_id}-{i}"))

            payload = {
                "doc_id": doc_id,
                "opd": opd,
                "filename": progress["filename"],
                "page_number": page_number,
                "chunk_index": i,
                "text": ch,
                "created_at": datetime.utcnow().isoformat()
            }

            points.append({
                "id": qpoint_id,
                "vector": vec,
                "payload": payload
            })

            progress["chunks"].append({
                "index": i,
                "page_number": page_number,
                "text_preview": ch[:120].replace("\n", " "),
                "duration_sec": round(chunk_dur, 3),
                "status": "embedded"
            })
            progress["total_chunks"] = len(progress["chunks"])

            # logging per chunk so tail -f terlihat progress
            logger.info(f"[DOC][chunk {i}] embedded (page={page_number}) in {chunk_dur:.3f}s, id={qpoint_id}")

        # ---------- ensure collection and upsert ----------
        t_ensure_start = time.time()
        size = len(points[0]["vector"])
        _ensure_collection(qdrant, collection_name, size)
        progress["stages"]["ensure_collection_sec"] = round(time.time() - t_ensure_start, 3)

        t_upsert_start = time.time()
        # upsert in batches to avoid too large payloads
        BATCH = 128
        for j in range(0, len(points), BATCH):
            batch = points[j: j + BATCH]
            qdrant.upsert(collection_name=collection_name, points=batch)
            logger.info(f"[DOC] Upsert batch {j // BATCH + 1} / {((len(points)-1)//BATCH)+1} ({len(batch)} points)")

        progress["stages"]["upsert_sec"] = round(time.time() - t_upsert_start, 3)
        total_dur = time.time() - t0
        progress["stages"]["total_sec"] = round(total_dur, 3)

        logger.info(f"[DOC] Selesai proses dokumen doc_id={doc_id} total_chunks={len(points)} total_time={total_dur:.3f}s")
        return {"status": "ok", "total_chunks": len(points), "filename": progress["filename"],
                "duration_sec": round(total_dur, 3), "progress": progress}

    except Exception as e:
        logger.exception("[DOC] Error saat proses dokumen")
        return {"status": "error", "message": str(e), "progress": progress}

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def _download_temp(url: str) -> str:
    """Download file dari URL ke temporary directory."""
    # requests tidak support file:// -> handle lokal path
    if url.startswith("file://"):
        local_path = url[len("file://"):]
        if not os.path.exists(local_path):
            raise ValueError(f"Local file not found: {local_path}")
        return local_path

    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    suffix = Path(url).suffix or ".bin"
    fd, tmp = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(resp.content)
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
            )
        )
