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

def detect_category(question):
    question_lower = question.lower()
    for category_id, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in question_lower for keyword in keywords):
            return {"id": category_id, "name": CATEGORY_NAMES[category_id]}
    return None


def normalize_text(text):
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def clean_location_terms(text):
    text = re.sub(r"\bdi\s+kota\s+medan\b", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bdi\s+medan\b", "", text, flags=re.IGNORECASE)
    return text.strip()


def expand_terms(text):
    words = text.lower().split()
    expanded_words = []
    for word in words:
        expanded_words.append(word)
        if word in SYNONYMS:
            expanded_words.extend(SYNONYMS[word])
    return " ".join(expanded_words)


def tokenize_and_filter(text):
    return [word.lower() for word in text.split() if word.lower() not in STOPWORDS and len(word) > 2]


def keyword_overlap(question_a, question_b):
    question_a_expanded = expand_terms(question_a)
    question_b_expanded = expand_terms(question_b)
    tokens_a = set(tokenize_and_filter(question_a_expanded))
    tokens_b = set(tokenize_and_filter(question_b_expanded))
    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b) if tokens_a and tokens_b else 0.0


def hard_filter_local(question: str):
    question_lower = question.lower()
    question_normalized = re.sub(r"[^\w\s]", " ", question_lower)
    question_normalized = re.sub(r"\s+", " ", question_normalized)

    NON_MEDAN_LOCATIONS = [
        "jakarta", "bandung", "surabaya", "yogyakarta", "semarang",
        "siantar", "pematangsiantar", "pematang siantar",
        "binjai", "tebing", "tebing tinggi", "aceh", "padang",
        "pekanbaru", "riau", "deliserdang", "deli serdang",
        "langkat", "tanjung morawa", "belawan", "labuhanbatu"
    ]

    OPINION_WORDS = [
        "rajin", "malas", "ganteng", "cantik", "baik", "buruk",
        "terkenal", "paling", "ter", "terbaik", "terburuk",
        "terjelek", "terbodoh", "terrajin"
    ]

    for location in NON_MEDAN_LOCATIONS:
        if re.search(rf"\b{re.escape(location)}\b", question_normalized):
            return {
                "valid": False,
                "reason": f"Pertanyaan menyebut daerah di luar Medan ({location.title()})",
                "clean_question": question
            }

    if any(re.search(rf"\b{re.escape(word)}\b", question_normalized) for word in OPINION_WORDS):
        return {
            "valid": False,
            "reason": "Pertanyaan bersifat opini/personal, bukan layanan publik",
            "clean_question": question
        }

    if len(question_normalized.split()) <= 1:
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


def safe_parse_answer_id(raw_value):
    """
    Pastikan answer_id selalu berbentuk list Python bersih (tanpa escape string).
    """
    if not raw_value:
        return []

    # Jika sudah list Python → kembalikan apa adanya
    if isinstance(raw_value, list):
        clean_list = []
        for item in raw_value:
            # Jika item seperti "\"abc\"" → decode
            try:
                if isinstance(item, str) and item.startswith('"') and item.endswith('"'):
                    clean_list.append(json.loads(item))
                else:
                    clean_list.append(item)
            except Exception:
                clean_list.append(item)
        return clean_list

    # Jika masih string → coba convert
    try:
        raw_string = str(raw_value).strip()

        # Case: '["\"uuid\""]'
        if raw_string.startswith("[") and raw_string.endswith("]"):
            parsed_array = ast.literal_eval(raw_string)
            clean_list = []
            for item in parsed_array:
                try:
                    clean_list.append(json.loads(item) if isinstance(item, str) and item.startswith('"') else item)
                except Exception:
                    clean_list.append(item)
            return clean_list

        # fallback → bungkus sebagai list
        return [raw_string]

    except Exception:
        return [str(raw_value)]
