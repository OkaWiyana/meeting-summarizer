"""
Auto-Summarizer untuk Dataset Anotasi DPR
==========================================
Cara pakai:
  1. pip install google-genai pandas
  2. Pastikan API key sudah diisi
  3. Jalankan: python auto_summarize.py
"""

import pandas as pd
from google import genai
import time
import re
from pathlib import Path

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
GEMINI_API_KEY      = "AIzaSyDtZNMHUx26tWnirFL6eagDwWYT8kvaYpI"
MODEL_NAME          = "gemini-2.5-flash"

PATH_CSV = Path(r"D:\Skripsi\meeting-summarizer\dataset\data_segment_siap_anotasi.csv")
DIR_OCR  = Path(r"D:\Skripsi\meeting-summarizer\dataset\02_extracted\ocr_risalah")

DELAY_ANTAR_REQUEST = 4   # 4 detik → aman untuk limit 20 req/menit
SIMPAN_TIAP_N_BARIS = 10
# ─────────────────────────────────────────


def buat_prompt(transkrip: str, referensi_ocr: str) -> str:
    return f"""Kamu adalah Asisten Peneliti NLP. 
Tugasmu: Buatkan ringkasan formal. Panjangnya sesuaikan dengan kepadatan informasi (kisaran 50-90 kata). Yang penting semua poin keputusan/argumen utama tidak hilang dari [TRANSKRIP LISAN] di bawah ini.
Carilah konteks pembahasannya di dalam [REFERENSI RISALAH UTUH] agar ringkasanmu sesuai fakta dan menggunakan gaya bahasa formal khas Risalah DPR.
PENTING: Hanya berikan hasil ringkasannya saja, tanpa basa-basi atau kalimat pengantar apapun!

[TRANSKRIP LISAN]:
{transkrip}

[REFERENSI RISALAH UTUH]:
{referensi_ocr}"""


def ringkas_satu_baris(client, transkrip: str, dokumen_asal: str) -> str:
    ocr_file = DIR_OCR / (dokumen_asal + ".txt")

    if ocr_file.exists():
        referensi_ocr = ocr_file.read_text(encoding="utf-8").strip()
    else:
        print(f"  ⚠  OCR tidak ditemukan untuk '{dokumen_asal}', referensi dikosongkan.")
        referensi_ocr = "(Referensi tidak tersedia)"

    prompt = buat_prompt(transkrip, referensi_ocr)

    maks_retry = 5
    for percobaan in range(maks_retry):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            return response.text.strip()

        except Exception as e:
            pesan = str(e)

            if "429" in pesan:
                # Baca retryDelay dari pesan error Gemini, lalu tunggu
                cocok = re.search(r'retryDelay.*?(\d+)s', pesan)
                tunggu = int(cocok.group(1)) + 5 if cocok else 65
                print(f"  ⏳ Rate limit! Tunggu {tunggu} detik... (percobaan {percobaan+1}/{maks_retry})")
                time.sleep(tunggu)

            elif "400" in pesan and "expired" in pesan:
                print("  ✗ API key expired! Perbarui key di aistudio.google.com lalu jalankan ulang.")
                raise

            else:
                print(f"  ✗ Error Gemini: {e}")
                return f"[ERROR: {e}]"

    return "[ERROR: Melebihi batas retry]"


def main():
    client = genai.Client(api_key=GEMINI_API_KEY)
    print(f"✓ Model siap: {MODEL_NAME}")

    df = pd.read_csv(PATH_CSV, encoding="utf-8", encoding_errors="replace")
    total = len(df)
    print(f"✓ CSV dimuat: {total} segmen")

    # Konversi ke string dulu (kolom kosong dibaca pandas sebagai float NaN)
    df["target_summary_manual"] = df["target_summary_manual"].fillna("").astype(str)

    # Proses baris yang kosong ATAU masih ERROR
    mask_belum = (df["target_summary_manual"].str.strip() == "") | \
                 (df["target_summary_manual"].str.startswith("[ERROR:"))
    indeks_belum = df[mask_belum].index.tolist()
    print(f"  → {len(indeks_belum)} baris belum diisi, {total - len(indeks_belum)} sudah ada ringkasan.\n")

    if not indeks_belum:
        print("Semua baris sudah terisi. Tidak ada yang perlu diproses.")
        return

    for i, idx in enumerate(indeks_belum, start=1):
        transkrip    = str(df.at[idx, "input_whisper_segment"])
        dokumen_asal = str(df.at[idx, "dokumen_asal"])

        print(f"[{i}/{len(indeks_belum)}] Memproses: {dokumen_asal} (baris {idx})...")

        ringkasan = ringkas_satu_baris(client, transkrip, dokumen_asal)
        df.at[idx, "target_summary_manual"] = ringkasan

        print(f"  ✓ {len(ringkasan.split())} kata → {ringkasan[:80]}...")

        if i % SIMPAN_TIAP_N_BARIS == 0:
            df.to_csv(PATH_CSV, index=False, encoding="utf-8")
            print(f"  💾 Auto-saved setelah {i} baris.")

        time.sleep(DELAY_ANTAR_REQUEST)

    df.to_csv(PATH_CSV, index=False, encoding="utf-8")
    print(f"\n✅ Selesai! CSV tersimpan di: {PATH_CSV}")
    print(f"   Total baris terisi: {(df['target_summary_manual'].str.strip() != '').sum()} / {total}")


if __name__ == "__main__":
    main()