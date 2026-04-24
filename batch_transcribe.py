"""
batch_transcribe.py
===================
Skrip batch untuk transkripsi otomatis semua file audio rapat sekaligus:
  - Scan folder: dataset/01_raw/video_mp4/, audio_mp3/, audio_wav/
  - MP4 → ekstrak audio ke WAV, lalu transkripsi
  - MP3 / WAV → transkripsi langsung
  - Hasil (.txt) disimpan ke: dataset/02_extracted/whisper_transcripts/
  - Nama file output sama dengan nama file input

Cara pakai (training — WAV saja):
    .\\.venv\\Scripts\\Activate.ps1
    python batch_transcribe.py

Atau langsung tanpa aktivasi:
    .\\.venv\\Scripts\\python.exe batch_transcribe.py
"""

import sys
import time
from pathlib import Path

# setup path
# ROOT = folder tempat file ini berada
ROOT = Path(__file__).parent 
# tambahkan root ke path agar bisa import modules
sys.path.insert(0, str(ROOT))

from modules.transcriber import transcribe_audio, WHISPER_MODEL, WHISPER_LANGUAGE
from modules.audio_utils import extract_audio_from_video
from modules.filter_kata import filter_teks

# konfigurasi path
DIR_MP4         = ROOT / "dataset" / "01_raw" / "video_mp4"
DIR_MP3         = ROOT / "dataset" / "01_raw" / "audio_mp3"
DIR_WAV         = ROOT / "dataset" / "01_raw" / "audio_wav"
DIR_OUTPUT      = ROOT / "dataset" / "02_extracted" / "whisper_transcripts"
DIR_TEMP_AUDIO  = ROOT / "data" / "temp_uploads"

# Untuk TRAINING menggunakan format ['wav']
# Untuk EVALUASI menggunakan format ['mp4', 'mp3', 'wav']
FORMAT_FILTER: list[str] = ['wav']

DIR_OUTPUT.mkdir(parents=True, exist_ok=True)
DIR_TEMP_AUDIO.mkdir(parents=True, exist_ok=True)


# fungsi mengumpulkan semua file audio dari folder raw
def kumpulkan_file_audio() -> list[tuple[Path, str]]:
    file_list = []
    if 'mp4' in FORMAT_FILTER:
        for f in sorted(DIR_MP4.glob("*.mp4")):
            file_list.append((f, "mp4"))
    if 'mp3' in FORMAT_FILTER:
        for f in sorted(DIR_MP3.glob("*.mp3")):
            file_list.append((f, "mp3"))
    if 'wav' in FORMAT_FILTER:
        for f in sorted(DIR_WAV.glob("*.wav")):
            file_list.append((f, "wav"))
    return file_list


# fungsi transkripsi satu file
def transkripsi_satu_file(audio_path: Path, fmt: str) -> tuple[str, float]:
    target_path = audio_path

    # untuk mp4 ekstrak audio terlebih dahulu
    if fmt == "mp4":
        print(f"Mengekstrak audio dari MP4...")
        target_path = extract_audio_from_video(audio_path, DIR_TEMP_AUDIO)

    t0 = time.time()
    teks = transcribe_audio(
        target_path,
        model_name=WHISPER_MODEL,
        language=WHISPER_LANGUAGE,
    )
    waktu = round(time.time() - t0, 2)
    return teks, waktu

# fungsi utama melakukan transkripsi
def main() -> None:
    print("=" * 60)
    print(f"  BATCH TRANSKRIPSI WHISPER — Model: {WHISPER_MODEL}")
    print(f"  Output: {DIR_OUTPUT}")
    print("=" * 60)

    file_list = kumpulkan_file_audio()

    if not file_list:
        print("\nTidak ada file audio ditemukan di folder raw.")
        print("   Pastikan file ada di:")
        print(f"   - {DIR_MP4}")
        print(f"   - {DIR_MP3}")
        print(f"   - {DIR_WAV}")
        return

    total = len(file_list)
    berhasil = 0
    gagal = []

    print(f"\nDitemukan {total} file audio. Memulai transkripsi...\n")

    for idx, (audio_path, fmt) in enumerate(file_list, start=1):
        out_path = DIR_OUTPUT / (audio_path.stem + ".txt")

        # Skip jika sudah pernah ditranskripsi
        if out_path.exists():
            print(f"[{idx:02d}/{total}] ⏭️  SKIP (sudah ada): {audio_path.name}")
            berhasil += 1
            continue

        print(f"[{idx:02d}/{total}] 🔊 {audio_path.name}  ({fmt.upper()})")

        try:
            teks, waktu = transkripsi_satu_file(audio_path, fmt)
            teks = filter_teks(teks, hapus=True)  # Hapus kata kasar dari transkrip berdasarkan kamus di module filter_kata.py
            out_path.write_text(teks, encoding="utf-8")
            kata_count = len(teks.split())
            print(f"Selesai dalam {waktu}s | {kata_count} kata → {out_path.name}\n")
            berhasil += 1

        except Exception as exc:
            print(f"GAGAL: {exc}\n")
            gagal.append(audio_path.name)

    # report hasil runing code
    print("=" * 60)
    print(f"  SELESAI: {berhasil}/{total} file berhasil ditranskripsi")
    if gagal:
        print(f"  GAGAL ({len(gagal)} file):")
        for f in gagal:
            print(f"    - {f}")
    print(f"  Hasil tersimpan di: {DIR_OUTPUT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
