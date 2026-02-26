"""
modules/filter_kata.py
======================
Modul pembersihan teks hasil transkripsi:
  - Filter kata kasar / tidak pantas
  - Normalisasi teks (spasi berlebih, karakter aneh, dll.)
  - Dapat dikembangkan dengan daftar kata kasar dari file eksternal
"""

import re
from pathlib import Path


# ── Daftar kata kasar (tambahkan sesuai kebutuhan) ───────────
# Dipisahkan dengan koma, gunakan huruf kecil semua.
# Untuk keperluan skripsi, isi ini dengan kata-kata yang relevan di konteks rapat.
KATA_KASAR: list[str] = [
    "anjing",
    "babi",
    "bangsat",
    "bodoh",
    "tolol",
    "goblok",
    "idiot",
    "kampret",
    "sialan",
    "keparat",
    # Tambahkan kata lain di sini ...
]

# Buat pola regex dari daftar kata kasar (case-insensitive, whole-word)
_PATTERN_KATA_KASAR = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in KATA_KASAR) + r")\b",
    flags=re.IGNORECASE,
)


def filter_kata_kasar(teks: str, pengganti: str = "***") -> str:
    """
    Ganti kata kasar dalam teks dengan karakter pengganti.

    Parameter:
        teks      : Teks input yang akan difilter.
        pengganti : String pengganti kata kasar. Default '***'.

    Return:
        Teks yang sudah difilter.
    """
    return _PATTERN_KATA_KASAR.sub(pengganti, teks)


def normalisasi_teks(teks: str) -> str:
    """
    Bersihkan dan normalisasi teks transkripsi.

    Langkah normalisasi:
      1. Hapus karakter non-ASCII yang tidak perlu (misal karakter kontrol)
      2. Normalisasi spasi ganda/tab menjadi spasi tunggal
      3. Hilangkan spasi di awal dan akhir baris
      4. Gabungkan baris kosong berulang menjadi satu baris kosong

    Parameter:
        teks : Teks input.

    Return:
        Teks yang sudah dinormalisasi.
    """
    # Hapus karakter kontrol (kecuali newline)
    teks = re.sub(r"[^\S\n]+", " ", teks)

    # Hilangkan spasi di awal/akhir setiap baris
    baris = [b.strip() for b in teks.splitlines()]

    # Gabungkan baris kosong berulang
    hasil = []
    baris_kosong_sebelumnya = False
    for b in baris:
        if b == "":
            if not baris_kosong_sebelumnya:
                hasil.append(b)
            baris_kosong_sebelumnya = True
        else:
            hasil.append(b)
            baris_kosong_sebelumnya = False

    return "\n".join(hasil).strip()


def filter_teks(teks: str) -> str:
    """
    Pipeline pembersihan teks lengkap:
      1. Filter kata kasar
      2. Normalisasi teks

    Parameter:
        teks : Teks transkripsi mentah dari Whisper.

    Return:
        Teks yang sudah bersih dan siap untuk summarization.
    """
    teks = filter_kata_kasar(teks)
    teks = normalisasi_teks(teks)
    return teks


def muat_kata_kasar_dari_file(path_file: Path) -> None:
    """
    Muat daftar kata kasar dari file teks eksternal (satu kata per baris)
    dan perbarui pola regex secara global.

    Parameter:
        path_file : Path ke file .txt berisi daftar kata kasar.
    """
    global KATA_KASAR, _PATTERN_KATA_KASAR

    path_file = Path(path_file)
    if not path_file.exists():
        raise FileNotFoundError(f"File kata kasar tidak ditemukan: {path_file}")

    with open(path_file, "r", encoding="utf-8") as f:
        kata_baru = [baris.strip().lower() for baris in f if baris.strip()]

    KATA_KASAR = kata_baru
    _PATTERN_KATA_KASAR = re.compile(
        r"\b(" + "|".join(re.escape(k) for k in KATA_KASAR) + r")\b",
        flags=re.IGNORECASE,
    )
