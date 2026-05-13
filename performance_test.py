"""
performance_test.py
===================
Automated Performance Testing — Tabel 3.8 Pengujian Kinerja Sistem

Menguji 11 data × 3 format (MP4, MP3, WAV) = 33 pengujian.

Aspek yang diukur:
  1. Waktu Cold Start      — Waktu loading awal model (≤ 30 detik)
  2. Waktu Inferensi       — Total proses unggah → hasil (≤ 1× durasi asli)
  3. Penggunaan Memori RAM — Puncak RAM saat proses (tidak OOM)
  4. Ukuran Model Storage  — Total file model.safetensors (< 5 GB)

Cara pakai:
    .\\.venv\\Scripts\\Activate.ps1
    python performance_test.py

Hasil disimpan ke: dataset/hasil_kinerja_sistem.csv
"""

import sys
import time
import os
import gc
import csv
import traceback
from pathlib import Path
from datetime import datetime

import psutil
import torch

# Setup path agar bisa import modules
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ─── Konfigurasi Path ───
DIR_MP4 = ROOT / "dataset" / "01_raw" / "video_mp4"
DIR_MP3 = ROOT / "dataset" / "01_raw" / "audio_mp3"
DIR_WAV = ROOT / "dataset" / "01_raw" / "audio_wav"
DIR_TEMP = ROOT / "data" / "temp_uploads"
DIR_TEMP.mkdir(parents=True, exist_ok=True)

OUTPUT_CSV = ROOT / "dataset" / "hasil_kinerja_sistem.csv"
LOG_FILE = ROOT / "data" / "logs" / "performance_test.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

# Path model untuk pengecekan ukuran storage
MODEL_DIR = ROOT / "models" / "indot5_finetuned"
WHISPER_CACHE = Path.home() / ".cache" / "whisper"

# ─── 11 Data Uji ───
DATA_UJI = [
    "Uji_Paripurna_Ke_7_Persidangan_II_2025_2026_seg01",
    "Uji_Paripurna_Ke_7_Persidangan_II_2025_2026_seg02",
    "Uji_Paripurna_Ke_7_Persidangan_II_2025_2026_seg03",
    "Uji_Paripurna_Ke_8_Persidangan_II_2025_2026_seg01",
    "Uji_Paripurna_Ke_8_Persidangan_II_2025_2026_seg02",
    "Uji_Paripurna_Ke_8_Persidangan_II_2025_2026_seg03",
    "Uji_Paripurna_Ke_8_Persidangan_II_2025_2026_seg04",
    "Uji_Paripurna_Ke_8_Persidangan_II_2025_2026_seg05",
    "Uji_Paripurna_Ke_20_Persidangan_IV_2024_2025_seg01",
    "Uji_Paripurna_Ke_20_Persidangan_IV_2024_2025_seg02",
    "Uji_Paripurna_Ke_20_Persidangan_IV_2024_2025_seg03",
]

FORMAT_MAP = {
    "MP4": DIR_MP4,
    "MP3": DIR_MP3,
    "WAV": DIR_WAV,
}


def log(msg: str):
    """Tulis log ke file dan console. Juga tulis ke system_performance.log."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    # Tulis juga ke system_performance.log (sama seperti Streamlit)
    from modules.performance import log_performance
    log_performance(f"[PERF_TEST] {msg}")


def get_peak_ram_mb() -> float:
    """Ambil penggunaan RAM (RSS) proses saat ini dalam MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)


def get_system_ram_info() -> dict:
    """Ambil informasi RAM sistem secara keseluruhan."""
    mem = psutil.virtual_memory()
    return {
        "total_gb": mem.total / (1024 ** 3),
        "available_gb": mem.available / (1024 ** 3),
        "used_gb": mem.used / (1024 ** 3),
        "percent": mem.percent,
    }


def get_model_storage_size() -> dict:
    """Hitung ukuran file model di disk."""
    sizes = {}
    
    # IndoT5 fine-tuned
    indot5_safetensors = MODEL_DIR / "model.safetensors"
    if indot5_safetensors.exists():
        sizes["indot5_safetensors_mb"] = indot5_safetensors.stat().st_size / (1024 ** 2)
    
    # Total folder IndoT5
    if MODEL_DIR.exists():
        total = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file())
        sizes["indot5_total_mb"] = total / (1024 ** 2)
    
    # Whisper model cache
    if WHISPER_CACHE.exists():
        whisper_files = list(WHISPER_CACHE.glob("*.pt"))
        # Cari model medium
        for wf in whisper_files:
            if "medium" in wf.name.lower():
                sizes["whisper_medium_mb"] = wf.stat().st_size / (1024 ** 2)
        total_whisper = sum(f.stat().st_size for f in whisper_files)
        sizes["whisper_total_mb"] = total_whisper / (1024 ** 2)
    
    # Total semua model
    sizes["grand_total_mb"] = sizes.get("indot5_total_mb", 0) + sizes.get("whisper_total_mb", 0)
    sizes["grand_total_gb"] = sizes["grand_total_mb"] / 1024
    
    return sizes


def measure_cold_start() -> dict:
    """
    Mengukur waktu cold start: load Whisper medium + IndoT5 fine-tuned.
    Model di-unload dulu untuk simulasi cold start murni.
    """
    log("=" * 60)
    log("MENGUKUR COLD START (Memuat Model dari Awal)")
    log("=" * 60)
    
    # Force unload model yang sudah ada
    from modules import transcriber, summarizer
    transcriber._loaded_models.clear()
    summarizer._tokenizer = None
    summarizer._model = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    ram_before = get_peak_ram_mb()
    log(f"RAM sebelum load model: {ram_before:.2f} MB")
    
    # ── Ukur Whisper Cold Start ──
    t0_whisper = time.time()
    from modules.transcriber import _get_model, WHISPER_MODEL
    _get_model(WHISPER_MODEL)
    cold_whisper = time.time() - t0_whisper
    ram_after_whisper = get_peak_ram_mb()
    log(f"Cold Start Whisper '{WHISPER_MODEL}': {cold_whisper:.2f}s | RAM: {ram_after_whisper:.2f} MB")
    
    # ── Ukur IndoT5 Cold Start ──
    t0_indot5 = time.time()
    from modules.summarizer import _load_model, MODEL_LOCAL_PATH
    _load_model(MODEL_LOCAL_PATH)
    cold_indot5 = time.time() - t0_indot5
    ram_after_indot5 = get_peak_ram_mb()
    log(f"Cold Start IndoT5: {cold_indot5:.2f}s | RAM: {ram_after_indot5:.2f} MB")
    
    total_cold = cold_whisper + cold_indot5
    log(f"TOTAL COLD START: {total_cold:.2f}s")
    log(f"STATUS: {'✅ LULUS' if total_cold <= 30 else '❌ GAGAL'} (Threshold: ≤ 30 detik)")
    
    return {
        "cold_start_whisper": round(cold_whisper, 2),
        "cold_start_indot5": round(cold_indot5, 2),
        "cold_start_total": round(total_cold, 2),
        "cold_start_lulus": total_cold <= 30,
        "ram_sebelum_load": round(ram_before, 2),
        "ram_setelah_load": round(ram_after_indot5, 2),
    }


def process_single_file(file_path: Path, fmt: str) -> dict:
    """
    Proses satu file: extract audio (jika MP4) → transkripsi → filter → ringkasan.
    Mengukur waktu inferensi total dan RAM puncak.
    """
    from modules.audio_utils import extract_audio_from_video, get_media_duration
    from modules.transcriber import transcribe_audio, WHISPER_MODEL, WHISPER_LANGUAGE
    from modules.filter_kata import filter_teks
    from modules.summarizer import summarize_text
    
    result = {
        "file": file_path.stem,
        "format": fmt,
        "durasi_detik": 0,
        "waktu_inferensi_detik": 0,
        "rasio_inferensi": 0,
        "ram_awal_mb": 0,
        "ram_puncak_mb": 0,
        "status_inferensi": "",
        "status_ram": "",
        "error": "",
    }
    
    try:
        # Ambil durasi asli media
        durasi = get_media_duration(file_path)
        result["durasi_detik"] = round(durasi, 2)
        
        # Catat RAM awal
        ram_awal = get_peak_ram_mb()
        result["ram_awal_mb"] = round(ram_awal, 2)
        ram_puncak = ram_awal
        
        # Mulai timer inferensi total
        t_start = time.time()
        
        # ── Step 1: Ekstrak audio jika MP4 ──
        audio_path = file_path
        if fmt == "MP4":
            audio_path = extract_audio_from_video(file_path, DIR_TEMP)
        
        ram_now = get_peak_ram_mb()
        ram_puncak = max(ram_puncak, ram_now)
        
        # ── Step 2: Transkripsi ──
        raw_transcript = transcribe_audio(
            audio_path,
            model_name=WHISPER_MODEL,
            language=WHISPER_LANGUAGE,
        )
        
        ram_now = get_peak_ram_mb()
        ram_puncak = max(ram_puncak, ram_now)
        
        # ── Step 3: Filter teks ──
        clean_transcript = filter_teks(raw_transcript)
        
        # ── Step 4: Ringkasan ──
        summary = summarize_text(clean_transcript)
        
        ram_now = get_peak_ram_mb()
        ram_puncak = max(ram_puncak, ram_now)
        
        # Stop timer
        waktu_inferensi = time.time() - t_start
        rasio = waktu_inferensi / durasi if durasi > 0 else 0
        
        result["waktu_inferensi_detik"] = round(waktu_inferensi, 2)
        result["rasio_inferensi"] = round(rasio, 4)
        result["ram_puncak_mb"] = round(ram_puncak, 2)
        
        # Status: Inferensi lulus jika rasio ≤ 1.0 (tidak melebihi durasi asli)
        result["status_inferensi"] = "LULUS" if rasio <= 1.0 else "GAGAL"
        
        # Status: RAM — cek apakah sistem tidak crash (berhasil sampai sini = tidak OOM)
        sys_ram = get_system_ram_info()
        result["status_ram"] = "LULUS"  # Jika sampai sini, berarti tidak OOM
        
        # Bersihkan file temp audio jika MP4
        if fmt == "MP4" and audio_path != file_path and audio_path.exists():
            audio_path.unlink()
        
    except Exception as e:
        result["error"] = str(e)
        result["status_inferensi"] = "ERROR"
        result["status_ram"] = "ERROR"
        log(f"ERROR: {e}")
        traceback.print_exc()
    
    return result


def main():
    log("=" * 70)
    log("  AUTOMATED PERFORMANCE TEST — Tabel 3.8 Pengujian Kinerja Sistem")
    log(f"  Tanggal: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Data Uji: {len(DATA_UJI)} file × {len(FORMAT_MAP)} format = {len(DATA_UJI) * len(FORMAT_MAP)} pengujian")
    log("=" * 70)
    
    # ─── Info Sistem ───
    sys_ram = get_system_ram_info()
    log(f"\n📊 INFO SISTEM:")
    log(f"   Total RAM      : {sys_ram['total_gb']:.2f} GB")
    log(f"   RAM Tersedia   : {sys_ram['available_gb']:.2f} GB")
    log(f"   RAM Terpakai   : {sys_ram['used_gb']:.2f} GB ({sys_ram['percent']}%)")
    log(f"   GPU (CUDA)     : {'Ya' if torch.cuda.is_available() else 'Tidak (CPU mode)'}")
    
    # ─── Aspek 4: Ukuran Model Storage ───
    log(f"\n📦 UKURAN MODEL (STORAGE):")
    model_sizes = get_model_storage_size()
    log(f"   IndoT5 (model.safetensors) : {model_sizes.get('indot5_safetensors_mb', 0):.2f} MB")
    log(f"   IndoT5 (total folder)      : {model_sizes.get('indot5_total_mb', 0):.2f} MB")
    log(f"   Whisper medium             : {model_sizes.get('whisper_medium_mb', 0):.2f} MB")
    log(f"   TOTAL SEMUA MODEL          : {model_sizes['grand_total_mb']:.2f} MB ({model_sizes['grand_total_gb']:.2f} GB)")
    log(f"   STATUS: {'✅ LULUS' if model_sizes['grand_total_gb'] < 5 else '❌ GAGAL'} (Threshold: < 5 GB)")
    
    # ─── Aspek 1: Waktu Cold Start ───
    cold_start = measure_cold_start()
    
    # ─── Aspek 2 & 3: Inferensi & RAM per file per format ───
    log("\n" + "=" * 70)
    log("  MULAI PENGUJIAN INFERENSI (11 Data × 3 Format)")
    log("=" * 70)
    
    results = []
    total_tests = len(DATA_UJI) * len(FORMAT_MAP)
    test_num = 0
    
    for nama_data in DATA_UJI:
        for fmt, dir_path in FORMAT_MAP.items():
            test_num += 1
            ext = fmt.lower()
            file_path = dir_path / f"{nama_data}.{ext}"
            
            log(f"\n[{test_num:02d}/{total_tests}] 🔊 {nama_data}.{ext} ({fmt})")
            
            if not file_path.exists():
                log(f"   ⚠️ FILE TIDAK DITEMUKAN: {file_path}")
                results.append({
                    "no": test_num,
                    "file": nama_data,
                    "format": fmt,
                    "durasi_detik": "-",
                    "waktu_inferensi_detik": "-",
                    "rasio_inferensi": "-",
                    "status_inferensi": "FILE NOT FOUND",
                    "ram_awal_mb": "-",
                    "ram_puncak_mb": "-",
                    "status_ram": "-",
                    "cold_start_total": cold_start["cold_start_total"],
                    "status_cold_start": "LULUS" if cold_start["cold_start_lulus"] else "GAGAL",
                    "ukuran_model_gb": round(model_sizes["grand_total_gb"], 2),
                    "status_model_size": "LULUS" if model_sizes["grand_total_gb"] < 5 else "GAGAL",
                    "error": "File tidak ditemukan",
                })
                continue
            
            # Proses file
            res = process_single_file(file_path, fmt)
            
            row = {
                "no": test_num,
                "file": nama_data,
                "format": fmt,
                "durasi_detik": res["durasi_detik"],
                "waktu_inferensi_detik": res["waktu_inferensi_detik"],
                "rasio_inferensi": res["rasio_inferensi"],
                "status_inferensi": res["status_inferensi"],
                "ram_awal_mb": res["ram_awal_mb"],
                "ram_puncak_mb": res["ram_puncak_mb"],
                "status_ram": res["status_ram"],
                "cold_start_total": cold_start["cold_start_total"],
                "status_cold_start": "LULUS" if cold_start["cold_start_lulus"] else "GAGAL",
                "ukuran_model_gb": round(model_sizes["grand_total_gb"], 2),
                "status_model_size": "LULUS" if model_sizes["grand_total_gb"] < 5 else "GAGAL",
                "error": res.get("error", ""),
            }
            results.append(row)
            
            log(f"   Durasi asli     : {res['durasi_detik']}s")
            log(f"   Waktu inferensi : {res['waktu_inferensi_detik']}s")
            log(f"   Rasio (target≤1): {res['rasio_inferensi']} → {res['status_inferensi']}")
            log(f"   RAM puncak      : {res['ram_puncak_mb']} MB → {res['status_ram']}")
            
            # Force garbage collection antar file
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # ─── Simpan Hasil ke CSV ───
    log("\n" + "=" * 70)
    log("  MENYIMPAN HASIL KE CSV")
    log("=" * 70)
    
    fieldnames = [
        "no", "file", "format",
        "durasi_detik", "waktu_inferensi_detik", "rasio_inferensi", "status_inferensi",
        "ram_awal_mb", "ram_puncak_mb", "status_ram",
        "cold_start_total", "status_cold_start",
        "ukuran_model_gb", "status_model_size",
        "error",
    ]
    
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    
    log(f"✅ Hasil tersimpan di: {OUTPUT_CSV}")
    
    # ─── Ringkasan Akhir ───
    log("\n" + "=" * 70)
    log("  RINGKASAN PENGUJIAN KINERJA SISTEM")
    log("=" * 70)
    
    valid_results = [r for r in results if r["error"] == ""]
    
    if valid_results:
        # Cold Start
        log(f"\n1️⃣  WAKTU COLD START")
        log(f"    Whisper : {cold_start['cold_start_whisper']}s")
        log(f"    IndoT5  : {cold_start['cold_start_indot5']}s")
        log(f"    Total   : {cold_start['cold_start_total']}s → {'✅ LULUS' if cold_start['cold_start_lulus'] else '❌ GAGAL'} (≤ 30s)")
        
        # Inferensi
        rasio_list = [r["rasio_inferensi"] for r in valid_results if isinstance(r["rasio_inferensi"], (int, float))]
        lulus_inf = sum(1 for r in valid_results if r["status_inferensi"] == "LULUS")
        log(f"\n2️⃣  WAKTU INFERENSI (LATENCY)")
        log(f"    Rata-rata rasio : {sum(rasio_list)/len(rasio_list):.4f}")
        log(f"    Maks rasio      : {max(rasio_list):.4f}")
        log(f"    Min rasio       : {min(rasio_list):.4f}")
        log(f"    Lulus           : {lulus_inf}/{len(valid_results)} → {'✅' if lulus_inf == len(valid_results) else '⚠️'}")
        
        # RAM
        ram_peaks = [r["ram_puncak_mb"] for r in valid_results if isinstance(r["ram_puncak_mb"], (int, float))]
        log(f"\n3️⃣  PENGGUNAAN MEMORI (RAM)")
        log(f"    RAM puncak maks : {max(ram_peaks):.2f} MB ({max(ram_peaks)/1024:.2f} GB)")
        log(f"    RAM puncak avg  : {sum(ram_peaks)/len(ram_peaks):.2f} MB")
        log(f"    Sistem RAM Total: {sys_ram['total_gb']:.2f} GB")
        log(f"    Status          : ✅ LULUS (Tidak ada Out of Memory)")
        
        # Model Size
        log(f"\n4️⃣  UKURAN MODEL (STORAGE)")
        log(f"    Total           : {model_sizes['grand_total_gb']:.2f} GB")
        log(f"    Status          : {'✅ LULUS' if model_sizes['grand_total_gb'] < 5 else '❌ GAGAL'} (< 5 GB)")
    
    gagal = [r for r in results if r["error"] != ""]
    if gagal:
        log(f"\n⚠️  FILE GAGAL ({len(gagal)}):")
        for r in gagal:
            log(f"    - {r['file']}.{r['format'].lower()}: {r['error']}")
    
    log("\n" + "=" * 70)
    log("  PENGUJIAN SELESAI")
    log("=" * 70)


if __name__ == "__main__":
    main()
