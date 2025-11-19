import re
import ast
import json

STOPWORDS = {
    "apa","bagaimana","cara","untuk","dan","atau","yang","dengan",
    "ke","dari","buat","membuat","mengurus","mendaftar","mencetak",
    "dimana","kapan","berapa","adalah","itu","ini","saya","kamu"
}

SYNONYMS = {
    "ktp": ["kartu tanda penduduk"],
    "kk": ["kartu keluarga"],
    "kadis": ["kepala dinas"],
    "kominfo": ["dinas komunikasi dan informatika", "diskominfo"],
    "dukcapil": ["dinas kependudukan dan catatan sipil", "disdukcapil"],
    "dishub": ["dinas perhubungan"],
    "dinkes": ["dinas kesehatan"],
    "disnaker": ["dinas ketenagakerjaan"],
    "sktm": ["surat keterangan tidak mampu"],
    "siup": ["surat izin usaha perdagangan"],
    "umkm": ["usaha mikro kecil menengah"],
    "pungli": ["pungutan liar"],
    "bansos": ["bantuan sosial"],
    "damkar": ["pemadam kebakaran"],
    "nib": ["nomor induk berusaha"],
    "nisn": ["nomor induk siswa nasional"],
    "pkl": ["praktek kerja lapangan"],
    "SKKNI": ["standar kompetensi kerja nasional indonesia"],
    "siduta": ["sistem informasi Terpadu Ketenagakerjaan"]
}

CATEGORY_KEYWORDS = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": [
        "ktp","kk","kartu keluarga","kartu tanda penduduk",
        "akta","kelahiran","kematian","domisili","SKTM","NIK"
    ],
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": [
        "bpjs","rsud","puskesmas","klinik","vaksin","pengobatan",
        "berobat","posyandu","stunting","imunisasi"
    ],
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": [
        "sekolah","PPDB","SPMB","guru","siswa","beasiswa","prestasi","zonasi","afirmasi","nisn"
    ],
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": [
        "pengaduan","izin","siup","bantuan","masyarakat","usaha","nib",
        "kartu prakerja", "kartu kuning", "AK1","sertifikat","pajak","reklame", "magang", "siduta"
    ],
    "0196f6b9-ba96-70f1-a930-3b89e763170f": [
        "kepala dinas","kadis","sekretaris","jabatan","struktur organisasi"
    ],
    "01970829-1054-72b2-bb31-16a34edd84fc": [
        "aturan","peraturan","perwali","perda","perpres","hukum"
    ],
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": [
        "lokasi","alamat","kantor","posisi"
    ],
    "001970853-dd2e-716e-b90c-c4f79270f700": [
        "tugas","fungsi","tupoksi","profil","visi","misi"
    ]
}

CATEGORY_NAMES = {
    "0196f6a8-9cb8-7385-8383-9d4f8fdcd396": "Kependudukan",
    "0196ccd1-d7f9-7252-b0a1-a67d4bc103a0": "Kesehatan",
    "0196cd16-3a0a-726d-99b4-2e9c6dda5f64": "Pendidikan",
    "019707b1-ebb6-708f-ad4d-bfc65d05f299": "Layanan Masyarakat",
    "0196f6b9-ba96-70f1-a930-3b89e763170f": "Struktur Organisasi",
    "01970829-1054-72b2-bb31-16a34edd84fc": "Peraturan",
    "0196f6c0-1178-733a-acd8-b8cb62eefe98": "Lokasi Fasilitas Pemerintahan Kota Medan",
    "001970853-dd2e-716e-b90c-c4f79270f700": "Profil"
}

ALL_KEYWORDS = set(sum(CATEGORY_KEYWORDS.values(), []))

def detect_category(q):
    ql = q.lower()
    for cid, kws in CATEGORY_KEYWORDS.items():
        if any(kw in ql for kw in kws):
            return {"id": cid, "name": CATEGORY_NAMES[cid]}
    return None


def normalize_text(t):
    t = re.sub(r"[^\w\s]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def clean_location_terms(t):
    t = re.sub(r"\bdi\s+kota\s+medan\b", "", t, flags=re.IGNORECASE)
    t = re.sub(r"\bdi\s+medan\b", "", t, flags=re.IGNORECASE)
    return t.strip()


def expand_terms(text):
    words = text.lower().split()
    expanded = []
    for w in words:
        expanded.append(w)
        if w in SYNONYMS:
            expanded.extend(SYNONYMS[w])
    return " ".join(expanded)


def tokenize_and_filter(t):
    return [w.lower() for w in t.split() if w.lower() not in STOPWORDS and len(w) > 2]


def keyword_overlap(a, b):
    a_exp = expand_terms(a)
    b_exp = expand_terms(b)
    A, B = set(tokenize_and_filter(a_exp)), set(tokenize_and_filter(b_exp))
    return len(A & B) / len(A | B) if A and B else 0.0


def hard_filter_local(question: str):
    q = question.lower()
    q_norm = re.sub(r"[^\w\s]", " ", q)
    q_norm = re.sub(r"\s+", " ", q_norm)

    NON_MEDAN = [
        "jakarta", "bandung", "surabaya", "yogyakarta", "semarang",
        "siantar", "pematangsiantar", "pematang siantar",
        "binjai", "tebing", "tebing tinggi", "aceh", "padang",
        "pekanbaru", "riau", "deliserdang", "deli serdang",
        "langkat", "tanjung morawa", "belawan", "labuhanbatu"
    ]

    OPINI_WORDS = [
        "rajin", "malas", "ganteng", "cantik", "baik", "buruk",
        "terkenal", "paling", "ter", "terbaik", "terburuk",
        "terjelek", "terbodoh", "terrajin"
    ]

    for city in NON_MEDAN:
        if re.search(rf"\b{re.escape(city)}\b", q_norm):
            return {
                "valid": False,
                "reason": f"Pertanyaan menyebut daerah di luar Medan ({city.title()})",
                "clean_question": question
            }

    if any(re.search(rf"\b{re.escape(w)}\b", q_norm) for w in OPINI_WORDS):
        return {
            "valid": False,
            "reason": "Pertanyaan bersifat opini/personal, bukan layanan publik",
            "clean_question": question
        }

    if len(q_norm.split()) <= 1:
        return {
            "valid": False,
            "reason": "Pertanyaan terlalu pendek atau tidak jelas",
            "clean_question": question
        }

    return {
        "valid": True,
        "reason": "Lolos hard filter",
        "clean_question": question
    }


def safe_parse_answer_id(raw):
    """
    Pastikan answer_id selalu berbentuk list Python bersih (tanpa escape string).
    """
    if not raw:
        return []

    # Jika sudah list Python → kembalikan apa adanya
    if isinstance(raw, list):
        clean = []
        for item in raw:
            # Jika item seperti "\"abc\"" → decode
            try:
                if isinstance(item, str) and item.startswith('"') and item.endswith('"'):
                    clean.append(json.loads(item))
                else:
                    clean.append(item)
            except Exception:
                clean.append(item)
        return clean

    # Jika masih string → coba convert
    try:
        raw_str = str(raw).strip()

        # Case: '["\"uuid\""]'
        if raw_str.startswith("[") and raw_str.endswith("]"):
            arr = ast.literal_eval(raw_str)
            clean = []
            for item in arr:
                try:
                    clean.append(json.loads(item) if isinstance(item, str) and item.startswith('"') else item)
                except Exception:
                    clean.append(item)
            return clean

        # fallback → bungkus sebagai list
        return [raw_str]

    except Exception:
        return [str(raw)]
