import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Tuple
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from docx import Document

# Inisialisasi engine OCR (gunakan cache agar tidak reload tiap kali)
# NOTE: tetap sama signature & nama fungsi; internalnya ditingkatkan (multi-thread)
ocr_engine = PaddleOCR(lang='id', use_angle_cls=True, show_log=False)

def _ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR dari bytes gambar (utility internal)."""
    import tempfile
    from PIL import Image
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(img_bytes)
        tmp.flush()
        path = tmp.name
    try:
        res = ocr_engine.ocr(path)
        if res and len(res) > 0:
            return "\n".join([line[1][0] for line in res[0]])
        return ""
    finally:
        try:
            os.remove(path)
        except Exception:
            pass

def _clean_page_text(t: str) -> str:
    """Bersihkan header/footer sederhana, spasi dobel, nomor halaman, dsb."""
    import re
    if not t:
        return ""
    # hapus nomor halaman yang berdiri sendiri (mis: "12")
    t = re.sub(r"^\s*\d{1,4}\s*$", "", t, flags=re.MULTILINE)
    # rapikan spasi
    t = re.sub(r"[ \t]+", " ", t)
    # gabungkan baris yang putus di tengah
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def _extract_pdf_pages(pdf_path: str, max_workers: int = 6, dpi: int = 200) -> Dict[int, str]:
    """
    Ekstrak teks PDF per halaman dengan hybrid:
    - Jika halaman punya teks vector → ambil get_text
    - Jika halaman image-scan → render → PaddleOCR
    Jalankan paralel untuk percepatan.
    """
    pages: Dict[int, str] = {}
    doc = fitz.open(pdf_path)

    def process_page(page_num: int) -> Tuple[int, str]:
        page = doc[page_num - 1]
        # jika ada teks native PDF, gunakan langsung
        text = page.get_text("text").strip()
        if text:
            return page_num, _clean_page_text(text)

        # jika tidak ada teks → OCR dari bitmap
        pix = page.get_pixmap(dpi=dpi)
        bytes_ = pix.tobytes("png")
        text = _ocr_image_bytes(bytes_)
        return page_num, _clean_page_text(text)

    # paralel
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(process_page, i) for i in range(1, len(doc) + 1)]
        for fu in as_completed(futures):
            p, t = fu.result()
            pages[p] = t

    # jaga urutan
    return dict(sorted(pages.items(), key=lambda x: x[0]))

def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Ekstraksi teks dari file PDF, DOCX, atau gambar.
    Jika return_pages=True, maka return (full_text, total_pages)
    Jika False, hanya return full_text saja.

    NOTE:
    - Signature tidak diubah.
    - Internal kini multi-thread untuk PDF scan.
    """
    ext = os.path.splitext(file_path)[1].lower()
    pages = {}

    if ext == ".pdf":
        pages = _extract_pdf_pages(file_path, max_workers=6, dpi=200)

    elif ext in [".jpg", ".jpeg", ".png"]:
        # OCR single image
        res = ocr_engine.ocr(file_path)
        text = ""
        if res and len(res) > 0:
            text = "\n".join([line[1][0] for line in res[0]])
        pages[1] = _clean_page_text(text)

    elif ext == ".docx":
        doc = Document(file_path)
        text = "\n".join([p.text.strip() for p in doc.paragraphs if p.text.strip()])
        pages[1] = _clean_page_text(text)

    else:
        raise ValueError(f"Format file {ext} belum didukung untuk OCR.")

    full_text = "\n\n".join(pages.values()).strip()

    if return_pages:
        return full_text, len(pages)
    return full_text
