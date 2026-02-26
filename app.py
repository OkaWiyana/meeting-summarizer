"""
app.py — Antarmuka utama Streamlit untuk Meeting Summarizer
============================================================
Alur utama:
  1. Pengguna upload file (MP4 / MP3 / WAV)
  2. Jika MP4 → ekstrak audio terlebih dahulu
  3. Transkripsi dengan Whisper (lokal)
  4. Filter kata kasar
  5. Ringkas dengan IndoT5 fine-tuned
  6. Tampilkan hasil & beri opsi download (PDF / TXT)
"""

import streamlit as st
import torch
from pathlib import Path

from modules.audio_utils import extract_audio_from_video
from modules.transcriber import transcribe_audio, WHISPER_MODEL
from modules.filter_kata import filter_teks
from modules.summarizer import summarize_text
from modules.exporter import export_to_pdf, export_to_txt

# ── Konfigurasi halaman ──────────────────────────────────────
st.set_page_config(
    page_title="Meeting Summarizer — Skripsi",
    page_icon="🎙️",
    layout="wide",
)

# ── Direktori sementara ─────────────────────────────────────
UPLOAD_DIR = Path("data/temp_uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Header ───────────────────────────────────────────────────
st.title("🎙️ Meeting Summarizer")
st.caption("Transkripsi & Ringkasan Rapat Otomatis Berbahasa Indonesia • Offline")

# ── Peringatan CPU ───────────────────────────────────────────
_device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
_color  = "green" if torch.cuda.is_available() else "orange"
st.markdown(
    f"""<div style='padding:8px 14px;border-radius:6px;
    background:{'#1a3a1a' if torch.cuda.is_available() else '#3a2800'};
    border-left:4px solid {'#2ecc71' if torch.cuda.is_available() else '#f39c12'};font-size:0.9em'>
    ⚡ Berjalan di <b>{_device}</b>.
    {'Transkripsi akan berjalan cepat.' if torch.cuda.is_available() else
     '⚠️ <b>Jangan tutup terminal/tab browser</b> selama proses berjalan!<br>'
     'Estimasi waktu di CPU: ~1–2 menit per menit audio.'}
    </div>""",
    unsafe_allow_html=True,
)
st.divider()

# ── Upload file ──────────────────────────────────────────────
uploaded_file = st.file_uploader(
    "Upload file rapat kamu (MP4 / MP3 / WAV)",
    type=["mp4", "mp3", "wav"],
)

if uploaded_file is not None:
    # Simpan file ke disk sementara
    save_path = UPLOAD_DIR / uploaded_file.name
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"File berhasil diupload: **{uploaded_file.name}**")
    st.divider()

    # ── Pipeline ─────────────────────────────────────────────
    st.info(
        f"🤖 Model ASR: **Whisper `{WHISPER_MODEL}`** │ "
        "Summarizer: **IndoT5 fine-tuned** │ Chunking: otomatis oleh sistem",
        icon="ℹ️",
    )
    run_btn = st.button("🚀 Mulai Proses", use_container_width=True)

    if run_btn:
        audio_path = save_path

        # Langkah 1: Ekstrak audio jika MP4
        if save_path.suffix.lower() == ".mp4":
            with st.spinner("🎬 Mengekstrak audio dari video MP4..."):
                audio_path = extract_audio_from_video(save_path, UPLOAD_DIR)
            st.success("✅ Audio berhasil diekstrak.")

        # Langkah 2: Transkripsi Whisper
        st.markdown(f"**🔊 Transkripsi dengan Whisper `{WHISPER_MODEL}`...**")
        _pbar = st.progress(0, text="Memulai transkripsi...")

        def _update_pbar(pct: float) -> None:
            _pbar.progress(pct, text=f"Transkripsi: {int(pct * 100)}%")

        raw_transcript = transcribe_audio(
            audio_path,
            progress_callback=_update_pbar,
        )
        _pbar.progress(1.0, text="Transkripsi selesai ✅")
        st.success("✅ Transkripsi selesai.")

        # Langkah 3: Filter kata kasar
        with st.spinner("🧹 Memfilter kata tidak pantas..."):
            clean_transcript = filter_teks(raw_transcript)
        st.success("✅ Filter selesai.")

        # Langkah 4: Ringkasan (chunking otomatis oleh sistem)
        with st.spinner("📝 Meringkas teks dengan IndoT5 (chunking otomatis)..."):
            summary = summarize_text(clean_transcript)
        st.success("✅ Ringkasan selesai.")

        st.divider()

        # ── Tampilkan hasil ───────────────────────────────────
        tab_trans, tab_sum = st.tabs(["📄 Transkripsi", "📋 Ringkasan"])

        with tab_trans:
            st.text_area("Hasil Transkripsi (sudah difilter)", clean_transcript, height=300)

        with tab_sum:
            st.text_area("Hasil Ringkasan", summary, height=200)

        # ── Export ────────────────────────────────────────────
        st.subheader("💾 Unduh Hasil")
        out_stem = save_path.stem

        pdf_path = export_to_pdf(summary, clean_transcript, OUTPUT_DIR / f"{out_stem}_hasil.pdf")
        txt_path = export_to_txt(summary, clean_transcript, OUTPUT_DIR / f"{out_stem}_hasil.txt")

        col_pdf, col_txt = st.columns(2)
        with col_pdf:
            with open(pdf_path, "rb") as f:
                st.download_button("⬇️ Unduh PDF", f, file_name=pdf_path.name, mime="application/pdf")
        with col_txt:
            with open(txt_path, "r", encoding="utf-8") as f:
                st.download_button("⬇️ Unduh TXT", f.read(), file_name=txt_path.name, mime="text/plain")
