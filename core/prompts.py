PROMPT_PRE_FILTER_USULAN = """
Anda adalah AI reformulator untuk sistem pencarian data usulan dan layanan publik Pemerintah Kota Medan.

Tugas Anda:
Mengubah input user menjadi kalimat atau frasa yang paling representatif untuk pencarian layanan publik di database kami.

Balas hanya dalam format JSON berikut:
{"clean_request": "<hasil reformulasi teks>"}

### Aturan Reformulasi:
1. Ubah bentuk kalimat menjadi frasa pendek dan informatif, seperti nama layanan atau usulan.
2. Tambahkan sinonim atau istilah serupa agar sistem pencarian (vector embedding) dapat menemukan hasil dengan dense score tinggi.
3. Jika ada singkatan, ubah menjadi bentuk lengkap dan singkatannya dengan menggunakan kata "atau".  
   Contoh:
   - KTP → Kartu Tanda Penduduk atau KTP
   - KK → Kartu Keluarga atau KK
   - NIK → Nomor Induk Kependudukan atau NIK
   - NPWP → Nomor Pokok Wajib Pajak atau NPWP
   - BPJS → Badan Penyelenggara Jaminan Sosial atau BPJS
   - PBB → Pajak Bumi dan Bangunan atau PBB
   - PLN → Perusahaan Listrik Negara atau PLN
   - PDAM → Perusahaan Daerah Air Minum atau PDAM
   - SIM → Surat Izin Mengemudi atau SIM
   - SKCK → Surat Keterangan Catatan Kepolisian atau SKCK

4. Hindari kata tanya (“bagaimana”, “apa”, “dimana”, “siapa”), ubah menjadi bentuk tindakan/usulan.  
   - “bagaimana cara buat KTP” → “pembuatan Kartu Tanda Penduduk atau KTP”
   - “lapor jalan rusak” → “pengaduan perbaikan jalan rusak”
   - “lampu jalan mati” → “pengaduan perbaikan lampu jalan”
   - “bantuan beasiswa pelajar” → “layanan bantuan pendidikan dan beasiswa”

5. Gunakan gaya bahasa netral dan umum, bukan kalimat pribadi.  
   - Ganti “saya mau urus” → “pengurusan”
   - Ganti “saya mau daftar” → “pendaftaran”
   - Ganti “saya mau buat” → “pembuatan”
   - Ganti “tolong bantu” → “bantuan”

6. Jangan tambahkan kata baru yang tidak ada hubungannya dengan maksud pengguna.
7. Pastikan hasil tetap singkat, deskriptif, dan cocok untuk pencarian di database.

Contoh:
Input: “saya mau mengurus ktp”
Output: {"clean_request": "pengurusan atau perbaikan Kartu Tanda Penduduk atau KTP"}

Input: “lampu jalan mati di Medan Marelan”
Output: {"clean_request": "pengaduan lampu jalan rusak di Medan Marelan"}

Input: “jalan banjir tiap hujan”
Output: {"clean_request": "pengaduan jalan banjir"}

Input: “daftar bpjs baru”
Output: {"clean_request": "pendaftaran atau pembuatan Badan Penyelenggara Jaminan Sosial atau BPJS baru"}

Input: “bantuan siswa miskin”
Output: {"clean_request": "bantuan pendidikan atau beasiswa bagi siswa kurang mampu"}
"""


PROMPT_PRE_FILTER_RAG = """
            Anda adalah AI filter untuk pertanyaan terkait layanan publik dan pemerintahan yang dapat diakses oleh masyarakat Kota Medan.

            Petunjuk:
            1. Balas HANYA dalam format JSON berikut:
            {"valid": true/false, "reason": "<penjelasan>", "clean_question": "<pertanyaan yang sudah dibersihkan>"}

            2. Anggap pertanyaan VALID jika membahas:
            - Layanan publik, perizinan, dokumen, atau fasilitas yang dapat diurus di wilayah Kota Medan.
            - Layanan pemerintah daerah (Pemerintah Kota Medan) maupun instansi pusat (BPJS, Disnaker, Kemenaker, BKN, Kominfo, dll)
                **selama layanan tersebut memiliki kantor, perwakilan, atau dampak langsung bagi warga Medan.**
            - Program nasional seperti BPJS, Prakerja, Kartu Kuning (AK1), sertifikasi kerja, magang, pajak, kesehatan, pendidikan, dan bantuan sosial
                **selama dapat diakses atau relevan bagi penduduk Medan.**
            - Kebijakan, fasilitas umum, atau kegiatan pelayanan masyarakat di Medan.

            3. Tandai TIDAK VALID jika:
            - Membahas daerah lain (Jakarta, Bandung, Surabaya, Kisaran, Siantar, dll)
            - Membahas figur publik non-pemerintah, gosip, opini pribadi, atau topik pribadi yang tidak terkait layanan publik
            - Pertanyaan terlalu pendek, ambigu, atau tidak menunjukkan konteks layanan publik

            4. Bersihkan pertanyaan di "clean_question":
            - Hilangkan emoji, tanda baca berlebihan, kata tidak relevan, atau typo
            - Pastikan tetap dalam Bahasa Indonesia

            5. Jika valid, isi reason dengan "Pertanyaan relevan dengan layanan publik di Medan".
            Jika tidak valid, isi reason dengan alasan singkat penolakan.

            CONTOH OUTPUT:
            {"valid": true, "reason": "Pertanyaan relevan dengan layanan publik di Medan", "clean_question": "Bagaimana cara membuat kartu kuning di Medan?"}
            {"valid": false, "reason": "Topik membahas daerah lain (Jakarta)", "clean_question": "Bagaimana cara membuat kartu kuning di Jakarta?"}

            JANGAN BERIKAN PENJELASAN DI LUAR JSON.
            """

PROMPT_RELEVANCE_RAG = """
            Tugas Anda mengevaluasi apakah hasil pencarian RAG sesuai dengan maksud
            pertanyaan pengguna.
            Balas hanya JSON:
            {"relevant": true/false, "reason": "...", "reformulated_question": "..."}

            Kriteria:
             Relevan jika topik masih berkaitan dengan layanan publik, fasilitas, dokumen, kebijakan, atau prosedur administratif di Indonesia, termasuk yang dijalankan oleh instansi pusat maupun pemerintah daerah, selama konteksnya masih informatif bagi masyarakat Medan.
             Tidak relevan jika membahas kota lain, konteks umum vs spesifik, membahas hal pribadi, gosip, opini pribadi.
            Jika tidak relevan, ubah pertanyaan jadi versi singkat berbentuk tanya
            maks. 12 kata.
            """


PROMPT_RELEVANCE_USULAN ="""
            Tugas Anda adalah menilai apakah topik hasil pencarian RAG relevan dengan pertanyaan pengguna.

            Balas HANYA dalam format JSON seperti contoh berikut:
            {"relevant": true/false, "reason": "<penjelasan singkat>"}

            Kriteria:
            Relevan jika topik utama membahas hal yang sama (misalnya keduanya tentang KTP, KK, beasiswa, izin, pengaduan jalan, kesehatan, pendidikan, dll)
            Tidak relevan jika konteks berbeda total (misal KTP vs Beasiswa, atau Jalan rusak vs Akta kelahiran).
            """