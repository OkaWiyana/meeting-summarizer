"""
modules/summarizer.py
=====================
Modul ringkasan teks menggunakan model IndoT5 fine-tuned.

Model disimpan di folder: models/indot5_finetuned/
Jika model lokal tidak tersedia, fallback ke model IndoT5 dari HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ── Konfigurasi sistem (tidak diubah oleh pengguna) ─────────
MODEL_LOCAL_PATH   = Path("models/indot5_finetuned")
# Model T5 pre-trained untuk Bahasa Indonesia
MODEL_HUB_FALLBACK = "Wikidepia/IndoT5-base"

# Batas token input maksimal untuk IndoT5 (sesuai arsitektur model)
MAX_INPUT_TOKENS  = 512   # token per chunk yang disuapkan ke model
# Panjang ringkasan per chunk (ditentukan sistem)
MAX_NEW_TOKENS    = 150
MIN_NEW_TOKENS    = 20
NUM_BEAMS         = 4
PREFIX_TASK       = "ringkas: "
# Ukuran chunk dalam kata (konservatif agar tidak melebihi MAX_INPUT_TOKENS)
CHUNK_SIZE_WORDS  = 300

# ── Cache agar model tidak di-load ulang ─────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None


def _load_model(model_path: Optional[Path] = None) -> tuple:
    """
    Load tokenizer dan model IndoT5.
    Prioritas: model lokal (fine-tuned) → model dari HuggingFace Hub.

    Return:
        Tuple (tokenizer, model)
    """
    global _tokenizer, _model

    if _tokenizer is not None and _model is not None:
        return _tokenizer, _model

    # Tentukan sumber model
    if model_path and model_path.exists():
        source = str(model_path)
    else:
        source = MODEL_HUB_FALLBACK

    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForSeq2SeqLM.from_pretrained(source)

    # Gunakan GPU jika tersedia
    device = "cuda" if torch.cuda.is_available() else "cpu"
    _model = _model.to(device)
    _model.eval()

    return _tokenizer, _model


def _ringkas_satu_chunk(teks_chunk: str, tokenizer, model, device) -> str:
    """
    Ringkas satu potongan teks menggunakan IndoT5.
    Fungsi internal — selalu menggunakan konstanta sistem.

    Parameter:
        teks_chunk : Potongan teks (maksimal CHUNK_SIZE_WORDS kata).
        tokenizer  : Tokenizer IndoT5.
        model      : Model IndoT5.
        device     : CPU/GPU device.

    Return:
        String ringkasan satu chunk.
    """
    input_text = PREFIX_TASK + teks_chunk
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_TOKENS,
        truncation=True,
        padding=False,
    ).to(device)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            min_length=MIN_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            early_stopping=True,
            no_repeat_ngram_size=3,
        )

    hasil = tokenizer.decode(output_ids[0], skip_special_tokens=True)

    # Bersihkan memori GPU setelah tiap chunk
    del inputs, output_ids
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return hasil.strip()


def summarize_text(teks: str) -> str:
    """
    Ringkas teks menggunakan IndoT5 fine-tuned.

    Chunking diterapkan SECARA OTOMATIS oleh sistem — teks dipecah
    menjadi potongan ≤ CHUNK_SIZE_WORDS kata, tiap potongan diringkas,
    lalu hasil digabungkan menjadi satu ringkasan akhir.
    Semua parameter (token length, beam, prefix) ditentukan sistem.

    Parameter:
        teks : Teks transkripsi yang sudah dibersihkan oleh filter_kata.

    Return:
        String ringkasan lengkap hasil model.
    """
    if not teks or not teks.strip():
        return ""

    tokenizer, model = _load_model(MODEL_LOCAL_PATH)
    device = next(model.parameters()).device

    # ── Chunking wajib: pecah teks berdasarkan batas kata ────
    kata = teks.split()
    potongan = [
        " ".join(kata[i : i + CHUNK_SIZE_WORDS])
        for i in range(0, len(kata), CHUNK_SIZE_WORDS)
    ]

    # Ringkas tiap chunk
    ringkasan_chunks = [
        _ringkas_satu_chunk(p, tokenizer, model, device)
        for p in potongan
    ]

    # Gabungkan ringkasan semua chunk
    gabungan = " ".join(ringkasan_chunks)

    # Jika gabungan masih terlalu panjang, lakukan satu putaran chunking lagi
    if len(gabungan.split()) > CHUNK_SIZE_WORDS:
        kata2 = gabungan.split()
        potongan2 = [
            " ".join(kata2[i : i + CHUNK_SIZE_WORDS])
            for i in range(0, len(kata2), CHUNK_SIZE_WORDS)
        ]
        ringkasan_final = " ".join(
            _ringkas_satu_chunk(p, tokenizer, model, device)
            for p in potongan2
        )
        return ringkasan_final.strip()

    return gabungan.strip()
