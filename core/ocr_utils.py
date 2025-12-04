import os
from typing import Dict, Tuple
from paddleocr import PaddleOCR
import fitz  # PyMuPDF
from docx import Document
import logging
import tempfile

logger = logging.getLogger("ocr_utils")

# ============================================================
# 🔹 Inisialisasi engine OCR (versi lama PaddleOCR 3.3.1)
#    ❗ Hanya parameter yang didukung: lang, use_angle_cls, show_log
# ============================================================
ocr_engine = PaddleOCR(
    lang="id",
    use_angle_cls=True
)

# ============================================================
# 🔹 Utility OCR untuk gambar (single-thread)
# ============================================================
def _ocr_image_bytes(img_bytes: bytes) -> str:
    """OCR dari bytes gambar (utility internal, tanpa multi-thread)."""
    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
            temp_file.write(img_bytes)
            temp_file.flush()
            temp_file_path = temp_file.name

        ocr_result = ocr_engine.ocr(temp_file_path)
        if ocr_result and len(ocr_result) > 0:
            return "\n".join([line[1][0] for line in ocr_result[0]])
        return ""
    except Exception as e:
        logger.warning(f"[OCR] Gagal OCR image: {e}")
        return ""
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except Exception:
                pass


# ============================================================
# 🔹 Utility pembersih teks halaman
# ============================================================
def _clean_page_text(page_text: str) -> str:
    """Bersihkan header/footer sederhana, spasi dobel, nomor halaman, dsb."""
    import re
    if not page_text:
        return ""
    # hapus nomor halaman yang berdiri sendiri, misal "12"
    cleaned_text = re.sub(r"^\s*\d{1,4}\s*$", "", page_text, flags=re.MULTILINE)
    # rapikan spasi
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    # gabungkan newline berlebih
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)
    return cleaned_text.strip()


# ============================================================
# 🔹 Ekstraksi PDF per halaman (HYBRID, TANPA THREAD)
# ============================================================
def _extract_pdf_pages(pdf_path: str, dpi: int = 180) -> Dict[int, str]:
    """
    Ekstrak teks PDF per halaman dengan hybrid:
    - Jika halaman punya teks vector → get_text
    - Jika halaman image-scan → render → PaddleOCR
    Diproses SEKUENSIAL (no multi-thread) demi stabilitas.
    """
    extracted_pages: Dict[int, str] = {}

    logger.info(f"[PDF] Membuka PDF: {pdf_path}")
    pdf_document = fitz.open(pdf_path)
    total_page_count = len(pdf_document)

    for page_number in range(1, total_page_count + 1):
        current_page = pdf_document[page_number - 1]

        # 1) Coba pakai teks bawaan PDF dulu
        try:
            extracted_text = current_page.get_text("text") or ""
        except Exception as e:
            logger.warning(f"[PDF] Gagal get_text di halaman {page_number}: {e}")
            extracted_text = ""

        extracted_text = extracted_text.strip()
        if extracted_text:
            extracted_pages[page_number] = _clean_page_text(extracted_text)
            continue

        # 2) Kalau tidak ada teks → OCR dari bitmap
        try:
            page_pixmap = current_page.get_pixmap(dpi=dpi)
            image_bytes = page_pixmap.tobytes("png")
            ocr_extracted_text = _ocr_image_bytes(image_bytes)
            extracted_pages[page_number] = _clean_page_text(ocr_extracted_text)
        except Exception as e:
            logger.warning(f"[PDF] Gagal render/OCR halaman {page_number}: {e}")
            extracted_pages[page_number] = ""

    return dict(sorted(extracted_pages.items(), key=lambda x: x[0]))


# ============================================================
# 🔹 Fungsi utama — Ekstraksi teks dari file
# ============================================================
def extract_text_from_file(file_path: str, lang: str = "id", return_pages: bool = False):
    """
    Ekstraksi teks dari file PDF, DOCX, atau gambar.
    Jika return_pages=True → kembalikan dict {page_number: text}
    Jika False → return string gabungan seluruh halaman.
    """
    file_extension = os.path.splitext(file_path)[1].lower()
    extracted_pages = {}

    if file_extension == ".pdf":
        extracted_pages = _extract_pdf_pages(file_path, dpi=180)

    elif file_extension in [".jpg", ".jpeg", ".png"]:
        try:
            ocr_result = ocr_engine.ocr(file_path)
        except Exception as e:
            logger.warning(f"[OCR] Gagal OCR image file {file_path}: {e}")
            ocr_result = None

        extracted_text = ""
        if ocr_result and len(ocr_result) > 0:
            extracted_text = "\n".join([line[1][0] for line in ocr_result[0]])
        extracted_pages[1] = _clean_page_text(extracted_text)

    elif file_extension == ".docx":
        docx_document = Document(file_path)
        extracted_text = "\n".join([paragraph.text.strip() for paragraph in docx_document.paragraphs if paragraph.text.strip()])
        extracted_pages[1] = _clean_page_text(extracted_text)

    else:
        raise ValueError(f"Format file {file_extension} belum didukung untuk OCR.")

    if return_pages:
        return dict(sorted(extracted_pages.items(), key=lambda x: x[0]))

    full_text = "\n\n".join(extracted_pages.values()).strip()
    return full_text
