"""
Auto-Summarizer untuk Dataset Anotasi DPR
==========================================
Cara pakai:
  1. pip install openai pandas python-dotenv rapidfuzz
  2. Daftar di https://console.groq.com → ambil API key gratis
  3. Isi key di file .env seperti ini:
       GROQ_KEY_1=gsk_xxxx
       GROQ_KEY_2=gsk_xxxx
       GROQ_KEY_3=gsk_xxxx
       dst...
  4. Jalankan: python auto_summarize.py
"""

import os
import pandas as pd
from openai import OpenAI
from rapidfuzz import fuzz
import time
import re
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────
# KONFIGURASI
# ─────────────────────────────────────────
# Baca semua key dari .env secara otomatis (GROQ_KEY_1, GROQ_KEY_2, dst)
API_KEYS = sorted([
    v for k, v in os.environ.items()
    if k.startswith("GROQ_KEY_") and v.strip()
])

MODEL_NAME          = "llama-3.3-70b-versatile"
PATH_CSV            = Path(r"D:\Skripsi\meeting-summarizer\dataset\data_segment_siap_anotasi.csv")
DIR_OCR             = Path(r"D:\Skripsi\meeting-summarizer\dataset\02_extracted\ocr_risalah")
DELAY_ANTAR_REQUEST = 5
SIMPAN_TIAP_N_BARIS = 10
MAKS_KATA_OCR       = 500
# ─────────────────────────────────────────


class GroqClientPool:
    """Kelola banyak API key Groq, otomatis pindah kalau satu key habis TPD."""

    def __init__(self, api_keys: list):
        if not api_keys:
            raise ValueError("Tidak ada API key! Pastikan .env sudah diisi dengan GROQ_KEY_1, GROQ_KEY_2, dst.")
        self.keys   = api_keys
        self.index  = 0
        self.client = self._buat_client(self.keys[0])
        print(f"✓ {len(self.keys)} API key dimuat")
        print(f"  → Menggunakan key #{self.index + 1}")

    def _buat_client(self, api_key: str) -> OpenAI:
        return OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )

    def next_key(self):
        self.index += 1
        if self.index >= len(self.keys):
            raise RuntimeError(
                f"🚫 Semua {len(self.keys)} API key sudah habis kuota hariannya!\n"
                f"   Tambah key baru di .env atau tunggu besok untuk reset kuota."
            )
        self.client = self._buat_client(self.keys[self.index])
        print(f"🔑 Pindah ke API key #{self.index + 1} dari {len(self.keys)}")

    def generate(self, prompt: str) -> str:
        maks_retry = 5
        for percobaan in range(maks_retry):
            try:
                response = self.client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.3,
                )
                return response.choices[0].message.content.strip()

            except Exception as e:
                pesan = str(e)

                if "429" in pesan or "rate" in pesan.lower():
                    # Cek apakah ini limit harian (TPD) atau per menit (TPM)
                    if "day" in pesan.lower() or "quota" in pesan.lower():
                        print(f"  ⚠  Key #{self.index + 1} habis kuota harian, pindah key...")
                        self.next_key()
                    else:
                        cocok = re.search(r'Please try again in ([\d.]+)s', pesan)
                        tunggu = float(cocok.group(1)) + 2 if cocok else 60
                        print(f"  ⏳ Rate limit! Tunggu {tunggu:.0f} detik... (percobaan {percobaan+1}/{maks_retry})")
                        time.sleep(tunggu)

                elif "401" in pesan or "invalid" in pesan.lower():
                    print(f"  ⚠  Key #{self.index + 1} tidak valid, pindah key...")
                    self.next_key()

                else:
                    print(f"  ✗ Error Groq: {e}")
                    return f"[ERROR: {e}]"

        return "[ERROR: Melebihi batas retry]"


def ambil_ocr_relevan(transkrip: str, referensi_ocr: str, maks_kata: int = 500) -> str:
    """Ambil paragraf OCR yang paling relevan dengan transkrip."""
    paragraf = [p.strip() for p in referensi_ocr.split('\n') if len(p.strip()) > 20]

    if not paragraf:
        return " ".join(referensi_ocr.split()[:maks_kata])

    sample = " ".join(transkrip.split()[:50])

    scored = [(fuzz.token_set_ratio(sample, p), p) for p in paragraf]
    scored.sort(reverse=True)

    hasil, total_kata = [], 0
    for _, p in scored:
        kata = p.split()
        if total_kata + len(kata) > maks_kata:
            break
        hasil.append(p)
        total_kata += len(kata)

    return " ".join(hasil) if hasil else " ".join(referensi_ocr.split()[:maks_kata])


def buat_prompt(transkrip: str, referensi_ocr: str) -> str:
    return f"""Kamu adalah Asisten Peneliti NLP yang bertugas membuat data training untuk model summarization IndoT5.

TUGAS: Buat ringkasan formal dari [TRANSKRIP LISAN] di bawah ini.

ATURAN WAJIB:
1. Panjang MAKSIMAL 4 kalimat, 50-70 kata — TIDAK BOLEH lebih
2. Langsung ke inti — TIDAK perlu sebut "Dewan Perwakilan Rakyat (DPR)" panjang-panjang, cukup "DPR RI" atau "Rapat Paripurna"
3. Gaya bahasa: singkat, padat, formal khas Risalah DPR — mirip berita resmi, bukan esai
4. Gunakan [REFERENSI RISALAH UTUH] HANYA untuk koreksi ejaan/nama, bukan tambah fakta baru
5. Sertakan angka/keputusan penting jika ada
6. JANGAN kembangkan singkatan yang tidak dijelaskan di transkrip — tulis apa adanya (BKN tetap BKN)
7. JANGAN ubah framing kejadian — jika anggota DPR menyampaikan aspirasi, tulis "Anggota DPR menyampaikan..." bukan "Rapat Paripurna menerima..."
8. JANGAN mengarang angka atau fakta yang tidak ada di transkrip
PENTING: Hanya berikan hasil ringkasannya saja, tanpa basa-basi atau kalimat pengantar apapun!

Contoh gaya yang BENAR (50-70 kata):
Rapat Paripurna DPR RI menyetujui RUU Undang-undang dan menyampaikan kebijakan negara pada Masa Persidangan II Tahun Sidang 2025-2026. Kebijakan pemerintah akan memberikan ruang untuk kesejahteraan rakyat serta menjaga keseimbangan ekonomi dan keuangan. Pemerintah juga mengapresiasi kinerja aparatur negara selama masa reses.

[TRANSKRIP LISAN]:
{transkrip}

[REFERENSI RISALAH UTUH]:
{referensi_ocr}"""


def ringkas_satu_baris(pool: GroqClientPool, transkrip: str, dokumen_asal: str) -> str:
    ocr_file = DIR_OCR / (dokumen_asal + ".txt")

    if ocr_file.exists():
        referensi_ocr_full = ocr_file.read_text(encoding="utf-8").strip()
        referensi_ocr = ambil_ocr_relevan(transkrip, referensi_ocr_full, MAKS_KATA_OCR)
    else:
        print(f"  ⚠  OCR tidak ditemukan untuk '{dokumen_asal}', referensi dikosongkan.")
        referensi_ocr = "(Referensi tidak tersedia)"

    prompt = buat_prompt(transkrip, referensi_ocr)
    return pool.generate(prompt)


def main():
    pool = GroqClientPool(API_KEYS)
    print(f"✓ Model siap: {MODEL_NAME}")

    df = pd.read_csv(PATH_CSV, encoding="utf-8", encoding_errors="replace")
    total = len(df)
    print(f"✓ CSV dimuat: {total} segmen")

    if "target_summary_manual" not in df.columns:
        df["target_summary_manual"] = ""
        print("  ℹ  Kolom 'target_summary_manual' dibuat baru.")

    df["target_summary_manual"] = df["target_summary_manual"].fillna("").astype(str)

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

        ringkasan = ringkas_satu_baris(pool, transkrip, dokumen_asal)
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