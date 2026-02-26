"""
modules/ocr_risalah.py
======================
Modul OCR untuk mengekstrak teks dari file Risalah Rapat (PDF):
  1. Konversi halaman PDF ke gambar (pdf2image + poppler)
  2. OCR tiap gambar dengan Tesseract (pytesseract)
  3. Bersihkan teks hasil OCR dengan regex
  4. Ekstrak bagian tertentu saja (antara kalimat/kata kunci A → B)
  5. Gabungkan (chunk) semua bagian menjadi 1 teks utuh untuk model

Dependensi eksternal:
  - Tesseract OCR terinstall di sistem dan tersedia di PATH
    Download: https://github.com/UB-Mannheim/tesseract/wiki
  - Poppler for Windows (untuk pdf2image)
    Download: https://github.com/oschwartz10612/poppler-windows/releases
"""

import re
from pathlib import Path

import pytesseract
from pdf2image import convert_from_path
from PIL import Image


# ── Konfigurasi sistem (tidak diubah pengguna) ───────────────
# Ubah path ini jika Tesseract tidak ada di PATH default.
# Contoh Windows: r"C:\Program Files\Tesseract-OCR\tesseract.exe"
TESSERACT_CMD: str | None = None  # None = pakai PATH default

# Konfigurasi OCR: bahasa Indonesia + English sebagai fallback
TESSERACT_LANG = "ind+eng"
TESSERACT_CONFIG = "--psm 6"  # Assume uniform block of text


# ── Pola default bagian yang diekstrak ───────────────────────
# Sesuaikan pasangan (pola_awal, pola_akhir) dengan format risalah kamu.
# Keduanya adalah pola regex (case-insensitive).
POLA_AWAL_DEFAULT = r"MENYANYIKAN LAGU INDONESIA RAYA"
POLA_AKHIR_DEFAULT = r"RAPAT DITUTUP PUKUL"


def _setup_tesseract() -> None:
    """Atur path Tesseract jika dikonfigurasi secara manual."""
    if TESSERACT_CMD:
        pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def pdf_ke_gambar(pdf_path: Path, dpi: int = 300) -> list[Image.Image]:
    """
    Konversi file PDF menjadi list gambar PIL per halaman.

    Parameter:
        pdf_path : Path ke file PDF.
        dpi      : Resolusi render (DPI). Default 300 untuk akurasi OCR optimal.

    Return:
        List objek PIL.Image, satu per halaman.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"File PDF tidak ditemukan: {pdf_path}")
    return convert_from_path(str(pdf_path), dpi=dpi)


def ocr_gambar(gambar: Image.Image) -> str:
    """
    Jalankan OCR pada satu gambar halaman.

    Parameter:
        gambar : Objek PIL.Image dari satu halaman PDF.

    Return:
        String teks hasil OCR mentah.
    """
    _setup_tesseract()
    return pytesseract.image_to_string(
        gambar,
        lang=TESSERACT_LANG,
        config=TESSERACT_CONFIG,
    )


def bersihkan_teks_ocr(teks: str) -> str:
    """
    Bersihkan artefak umum hasil OCR:
      - Hapus karakter aneh / non-ASCII berlebih
      - Normalisasi spasi dan newline ganda
      - Hapus nomor halaman pola umum (mis. "- 1 -", "Halaman 1")
      - Perbaiki kata yang terpotong di akhir baris karena hyphen

    Parameter:
        teks : Teks mentah dari OCR.

    Return:
        Teks yang sudah dibersihkan.
    """
    # Sambungkan kata yang terpotong hyphen di akhir baris
    teks = re.sub(r"-\n(\w)", r"\1", teks)

    # Hapus nomor halaman umum dalam berbagai format:
    # -34-  |  - 34 -  |  -iv-  |  Halaman 34  |  Page 34
    teks = re.sub(r"(?i)-\s*\d+\s*-", "", teks)               # -34- atau - 34 -
    teks = re.sub(r"(?i)-\s*[ivxlcdmIVXLCDM]+\s*-", "", teks) # -iv- (romawi)
    teks = re.sub(r"(?i)\bhalaman\s*\d+\b", "", teks)          # Halaman 34
    teks = re.sub(r"(?i)\bpage\s*\d+\b", "", teks)             # Page 34

    # Hapus karakter non-cetak kecuali newline dan spasi
    teks = re.sub(r"[^\x20-\x7EÀ-ÖØ-öø-ÿ\n]", " ", teks)

    # Normalisasi spasi berlebih dan newline ganda
    teks = re.sub(r"[ \t]+", " ", teks)
    teks = re.sub(r"\n{3,}", "\n\n", teks)

    # Hilangkan spasi di awal/akhir tiap baris
    baris = [b.strip() for b in teks.splitlines()]
    teks = "\n".join(baris).strip()

    return teks


def ekstrak_bagian(
    teks: str,
    pola_awal: str = POLA_AWAL_DEFAULT,
    pola_akhir: str = POLA_AKHIR_DEFAULT,
) -> str:
    """
    Ekstrak porsi teks di antara dua penanda (pola_awal → pola_akhir).

    Jika penanda tidak ditemukan, kembalikan seluruh teks sebagai fallback
    agar proses tidak berhenti total.

    Parameter:
        teks      : Teks hasil OCR yang sudah dibersihkan.
        pola_awal : Pola regex penanda awal bagian yang ingin diekstrak.
        pola_akhir: Pola regex penanda akhir bagian.

    Return:
        Teks bagian yang diekstrak, atau teks penuh jika pola tidak ditemukan.
    """
    pola = re.compile(
        rf"(?i)(?:{pola_awal})(.*?)(?:{pola_akhir})",
        flags=re.DOTALL | re.IGNORECASE,
    )
    hasil = pola.findall(teks)

    if hasil:
        blok_bersih = []
        for bagian in hasil:
            # Hapus kurung tutup ) di awal dan kurung buka ( di akhir
            # yang ikut terbawa karena format risalah: "ANCHOR_AWAL) ... (ANCHOR_AKHIR"
            bagian = re.sub(r"^\s*\)\s*", "", bagian)   # buang ) di awal
            bagian = re.sub(r"\s*\(\s*$", "", bagian)   # buang ( di akhir
            blok_bersih.append(bagian.strip())
        return "\n\n".join(blok_bersih)

    # Fallback: kembalikan teks penuh
    return teks


def proses_pdf_risalah(
    pdf_path: Path,
    pola_awal: str = POLA_AWAL_DEFAULT,
    pola_akhir: str = POLA_AKHIR_DEFAULT,
    simpan_ke: Path | None = None,
) -> str:
    """
    Pipeline lengkap: PDF → OCR → Bersihkan → Ekstrak Bagian → Teks Gabungan.

    Satu fungsi ini mencakup semua langkah sehingga bisa dipanggil langsung
    dari notebook atau skrip pelatihan.

    Parameter:
        pdf_path  : Path ke file PDF risalah rapat.
        pola_awal : Regex penanda awal bagian yang diekstrak. Default: kata kunci pembahasan.
        pola_akhir: Regex penanda akhir. Default: kata kunci penutup/kesimpulan.
        simpan_ke : Jika diisi, teks hasil akan disimpan ke path ini (.txt).

    Return:
        String teks gabungan siap pakai sebagai `target` (ground truth) model.
    """
    pdf_path = Path(pdf_path)

    # Langkah 1 & 2: PDF → Gambar → OCR tiap halaman
    halaman = pdf_ke_gambar(pdf_path)
    teks_per_halaman = [ocr_gambar(h) for h in halaman]
    teks_gabungan_mentah = "\n\n".join(teks_per_halaman)

    # Langkah 3: Bersihkan artefak OCR
    teks_bersih = bersihkan_teks_ocr(teks_gabungan_mentah)

    # Langkah 4: Ekstrak bagian relevan saja
    teks_final = ekstrak_bagian(teks_bersih, pola_awal, pola_akhir)

    # Opsional: simpan ke file .txt
    if simpan_ke:
        simpan_ke = Path(simpan_ke)
        simpan_ke.parent.mkdir(parents=True, exist_ok=True)
        simpan_ke.write_text(teks_final, encoding="utf-8")

    return teks_final


def proses_semua_pdf(
    direktori_pdf: Path,
    direktori_output: Path,
    pola_awal: str = POLA_AWAL_DEFAULT,
    pola_akhir: str = POLA_AKHIR_DEFAULT,
) -> dict[str, str]:
    """
    Proses semua file PDF dalam satu direktori secara batch.

    Parameter:
        direktori_pdf    : Folder berisi file PDF risalah.
        direktori_output : Folder tempat hasil .txt disimpan.
        pola_awal        : Regex penanda awal.
        pola_akhir       : Regex penanda akhir.

    Return:
        Dict {nama_file: teks_hasil} untuk semua PDF yang berhasil diproses.
    """
    direktori_pdf = Path(direktori_pdf)
    direktori_output = Path(direktori_output)
    direktori_output.mkdir(parents=True, exist_ok=True)

    hasil = {}
    for pdf_file in sorted(direktori_pdf.glob("*.pdf")):
        try:
            out_path = direktori_output / (pdf_file.stem + ".txt")
            teks = proses_pdf_risalah(pdf_file, pola_awal, pola_akhir, simpan_ke=out_path)
            hasil[pdf_file.name] = teks
            print(f"✅ {pdf_file.name} → {len(teks.split())} kata diekstrak")
        except Exception as exc:
            print(f"❌ Gagal memproses {pdf_file.name}: {exc}")

    return hasil
