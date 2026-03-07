"""
modules/summarizer.py
=====================
Modul ringkasan teks menggunakan model IndoT5 fine-tuned.

Model disimpan di folder: models/indot5_finetuned/
Jika model lokal tidak tersedia, fallback ke model IndoT5 dari HuggingFace Hub.
"""

from pathlib import Path
from typing import Optional
import re

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch


# ── Konfigurasi sistem  ─────────
MODEL_LOCAL_PATH   = Path("models/indot5_finetuned")
MODEL_HUB_FALLBACK = "Wikidepia/IndoT5-base"

# Batas token input maksimal (Sesuai Training)
MAX_INPUT_TOKENS  = 512   
# Panjang ringkasan per chunk
MAX_NEW_TOKENS    = 100   
MIN_NEW_TOKENS    = 20
NUM_BEAMS         = 4
PREFIX_TASK       = "ringkas: " 
REP_PENALTY       = 2.0   

# Ukuran chunk dalam kata
CHUNK_SIZE_WORDS  = 300

# ── Cache agar model tidak di-load ulang ─────────────────────
_tokenizer: Optional[AutoTokenizer] = None
_model: Optional[AutoModelForSeq2SeqLM] = None


def _load_model(model_path: Optional[Path] = None) -> tuple:
    """
    Load tokenizer dan model IndoT5.
    Prioritas: model lokal (fine-tuned) → model dari HuggingFace Hub.
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
    Ringkas satu potongan teks menggunakan IndoT5 dengan parameter yang
    SUDAH DISAMAKAN dengan pengujian evaluasi ROUGE.
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
            repetition_penalty=REP_PENALTY,
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
    if not teks or not teks.strip():
        return ""

    tokenizer, model = _load_model(MODEL_LOCAL_PATH)
    device = next(model.parameters()).device

    teks = teks.lower()
    teks = re.sub(r'^[^\w\s]+', '', teks)
    teks = re.sub(r'[^\x00-\x7F\u00C0-\u024F\s]', ' ', teks)
    teks = re.sub(r'[^\w\s.,!?-]', ' ', teks)
    teks = re.sub(r'\s+([.,!?])', r'\1', teks)
    teks = re.sub(r'(\d)\s*\.\s*(\d)', r'\1,\2', teks)
    teks = re.sub(r'\b(\w+)\s+\1\b', r'\1', teks)
    teks = re.sub(r'\s+', ' ', teks).strip()

    kalimat_list = re.split(r'(?<=[.!?]) +', teks)
    potongan = []
    chunk_saat_ini = []
    jumlah_kata_saat_ini = 0

    for kalimat in kalimat_list:
        jml_kata = len(kalimat.split())
        if jumlah_kata_saat_ini + jml_kata > 350 and jumlah_kata_saat_ini > 0:
            potongan.append(" ".join(chunk_saat_ini))
            chunk_saat_ini = [kalimat]
            jumlah_kata_saat_ini = jml_kata
        else:
            chunk_saat_ini.append(kalimat)
            jumlah_kata_saat_ini += jml_kata

    if chunk_saat_ini and jumlah_kata_saat_ini > 50:
        potongan.append(" ".join(chunk_saat_ini))

    ringkasan_chunks = [
        _ringkas_satu_chunk(p, tokenizer, model, device)
        for p in potongan
    ]

    return " ".join(ringkasan_chunks).strip()