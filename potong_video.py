"""
potong_video.py
================
Script untuk memotong video berdasarkan timestamp manual,
lalu otomatis menghasilkan 3 format: MP4, MP3, WAV.

Cara pakai:
  1. Isi VIDEO_ASLI  → nama file MP4 sumber (taruh di dataset/01_raw/video_mp4/)
  2. Isi NAMA_PREFIX → nama dasar untuk file output
  3. Isi CHUNKS      → list waktu potong (start, end) dalam format HH:MM:SS
  4. Jalankan:  python potong_video.py
"""

import os
import subprocess
from pathlib import Path

# ─────────────────────────────────────────────
#  ✏️  EDIT BAGIAN INI SESUAI KEBUTUHAN
# ─────────────────────────────────────────────

# Nama file video sumber (harus ada di folder dataset/01_raw/video_mp4/)
VIDEO_ASLI = "Paripurna_Ke_20_Persidangan_IV_2024_2025.mp4"

# Prefix nama file output (potongan akan diberi suffix _seg01, _seg02, dst.)
NAMA_PREFIX = "Uji_Paripurna_Ke_20_Persidangan_IV_2024_2025"

# Daftar potongan: isi start dan end dalam format HH:MM:SS
CHUNKS = [
    {"id": 1, "start": "00:00:00", "end": "00:03:20"},
    {"id": 2, "start": "00:03:20", "end": "00:06:43"},
    {"id": 3, "start": "00:06:43", "end": "00:10:52"},
]

# ─────────────────────────────────────────────
#  ⚙️  KONFIGURASI PATH (tidak perlu diubah)
# ─────────────────────────────────────────────

ROOT        = Path(__file__).parent
DIR_MP4     = ROOT / "dataset" / "01_raw" / "video_mp4"
DIR_MP3     = ROOT / "dataset" / "01_raw" / "audio_mp3"
DIR_WAV     = ROOT / "dataset" / "01_raw" / "audio_wav"
DIR_MP4.mkdir(parents=True, exist_ok=True)
DIR_MP3.mkdir(parents=True, exist_ok=True)
DIR_WAV.mkdir(parents=True, exist_ok=True)

VIDEO_PATH  = DIR_MP4 / VIDEO_ASLI

# ─────────────────────────────────────────────
#  🚀  PROSES PEMOTONGAN
# ─────────────────────────────────────────────

def run(cmd: list[str], label: str) -> bool:
    """Jalankan perintah ffmpeg dan tampilkan status."""
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"    ✅ {label}")
        return True
    else:
        print(f"    ❌ GAGAL: {label}")
        print(f"       {result.stderr.splitlines()[-1] if result.stderr else 'unknown error'}")
        return False


def main():
    if not VIDEO_PATH.exists():
        print(f"❌  File tidak ditemukan: {VIDEO_PATH}")
        return

    print("=" * 60)
    print(f"  PEMOTONGAN VIDEO: {VIDEO_ASLI}")
    print(f"  Jumlah potongan : {len(CHUNKS)}")
    print("=" * 60)

    for chunk in CHUNKS:
        idx   = chunk["id"]
        start = chunk["start"]
        end   = chunk["end"]
        nama  = f"{NAMA_PREFIX}_seg{idx:02d}"

        out_mp4 = DIR_MP4 / f"{nama}.mp4"
        out_mp3 = DIR_MP3 / f"{nama}.mp3"
        out_wav = DIR_WAV / f"{nama}.wav"

        print(f"\n[{idx}/{len(CHUNKS)}] {nama}  ({start} → {end})")

        # 1. Potong MP4 (-c copy = sangat cepat, tidak re-encode)
        run([
            "ffmpeg", "-y",
            "-i", str(VIDEO_PATH),
            "-ss", start, "-to", end,
            "-c", "copy",
            str(out_mp4)
        ], f"MP4 → {out_mp4.name}")

        # 2. Konversi ke MP3 (128k)
        run([
            "ffmpeg", "-y",
            "-i", str(out_mp4),
            "-vn", "-ar", "16000", "-ac", "1", "-ab", "128k",
            str(out_mp3)
        ], f"MP3 → {out_mp3.name}")

        # 3. Konversi ke WAV (16kHz mono, cocok untuk Whisper)
        run([
            "ffmpeg", "-y",
            "-i", str(out_mp4),
            "-vn", "-ar", "16000", "-ac", "1",
            str(out_wav)
        ], f"WAV → {out_wav.name}")

    print("\n" + "=" * 60)
    print("  SELESAI! File tersimpan di:")
    print(f"    MP4 → {DIR_MP4}")
    print(f"    MP3 → {DIR_MP3}")
    print(f"    WAV → {DIR_WAV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
