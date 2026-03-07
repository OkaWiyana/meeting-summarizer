"""
app.py — Antarmuka utama Streamlit untuk Meeting Summarizer
============================================================
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
    page_title="Meeting Summarizer",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────
st.markdown("""
<style>
    /* Mengatur jarak halaman agar pas */
    .block-container {
        padding-top: 2rem !important;
        max-width: 800px !important;
    }
    
    /* Tombol utama (Primary) lebih menonjol */
    .stButton>button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Sudut textarea yang lebih halus */
    .stTextArea textarea {
        border-radius: 10px !important;
        line-height: 1.6 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── Direktori sementara ──────────────────────────────────────
UPLOAD_DIR = Path("data/temp_uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = ["mp4", "mp3", "wav"]
MAX_DURATION_MINUTES = 40

# ── Inisialisasi Session State ───────────────────────────────
for key in ["raw_transcript", "clean_transcript", "summary",
            "save_path", "processed", "audio_path"]:
    if key not in st.session_state:
        st.session_state[key] = None

# ── SIDEBAR (Untuk Info Teknis) ──────────────────────────────
with st.sidebar:
    st.title("Sistem")
    st.caption("Informasi teknis mesin pemroses.")
    
    _device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        st.success(f"Akselerasi: **{_device}**", icon=":material/speed:")
    else:
        st.warning(f"Berjalan di: **{_device}**\n\nProses mungkin memakan waktu.", icon=":material/memory:")
        
    st.divider()
    st.markdown("**Spesifikasi Model:**")
    st.markdown(f"- **ASR:** Whisper `{WHISPER_MODEL}`\n- **NLP:** IndoT5\n- **Chunking:** Otomatis")
    
    # st.divider()
    # st.caption("💡 **Tip Mode Gelap:**\nKlik menu ⋮ di kanan atas > Settings > Theme > Pilih Dark/Light.")

# ── HALAMAN UTAMA (Fokus User) ───────────────────────────────
st.title("Meeting Summarizer")
st.markdown("Unggah rekaman rapat, dan sistem akan membuat transkripsi serta ringkasan secara otomatis.")

# ── STEP 1: UPLOAD ──
uploaded_file = st.file_uploader(
    "Pilih file rekaman (MP4, MP3, WAV)",
    type=ALLOWED_EXTENSIONS,
    label_visibility="collapsed"
)

if uploaded_file is not None:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    save_path = UPLOAD_DIR / uploaded_file.name

    if st.session_state["save_path"] != str(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["save_path"] = str(save_path)
        st.session_state["processed"] = False
        st.session_state["summary"] = None

    # ── STEP 2: PREVIEW & ACTION (Di dalam Card) ──
    with st.container(border=True):
        st.subheader("Detail File", anchor=False)
        st.markdown(f"**Nama file:** `{uploaded_file.name}`")
        
        # Audio/Video player mini
        if file_ext == "mp4":
            st.video(str(save_path))
        else:
            st.audio(str(save_path))

        # Tombol Proses Utama
        run_btn = st.button("Proses Rekaman Ini", use_container_width=True, type="primary", icon=":material/auto_awesome:")

    # ── STEP 3: PROSES BERJALAN ──
    if run_btn:
        st.session_state["processed"] = False
        audio_path = save_path

        # st.status membuat proses loading tersembunyi rapi dalam dropdown
        with st.status("Memproses rapat, mohon tunggu...", expanded=True) as status_box:
            
            if save_path.suffix.lower() == ".mp4":
                st.write("Mengekstrak audio dari video...")
                audio_path = extract_audio_from_video(save_path, UPLOAD_DIR)
            st.session_state["audio_path"] = str(audio_path)

            st.write("Mentranskripsi percakapan...")
            
            # Progress bar untuk transkripsi dengan ANGKA PERSENTASE
            _pbar = st.progress(0, text="Persiapan transkripsi: 0%")
            
            def _update_pbar(pct: float):
                # Memastikan nilai pct aman (0.0 sampai 1.0)
                safe_pct = max(0.0, min(pct, 1.0))
                _pbar.progress(safe_pct, text=f"Proses transkripsi: {int(safe_pct * 100)}%")

            raw_transcript = transcribe_audio(audio_path, progress_callback=_update_pbar)
            
            # Pastikan progress bar mentok 100% saat selesai
            _pbar.progress(1.0, text="Transkripsi selesai: 100%")
            st.session_state["raw_transcript"] = raw_transcript

            st.write("Membersihkan teks...")
            clean_transcript = filter_teks(raw_transcript)
            st.session_state["clean_transcript"] = clean_transcript

            st.write("Menyusun ringkasan utama...")
            summary = summarize_text(clean_transcript)
            st.session_state["summary"] = summary
            
            st.session_state["processed"] = True
            
            # Ubah status jadi selesai dan tutup dropdown-nya otomatis
            status_box.update(label="Pemrosesan Selesai!", state="complete", expanded=False)

    # ── STEP 4: HASIL & DOWNLOAD (Hanya muncul jika selesai) ──
    if st.session_state["processed"] and st.session_state["summary"]:
        
        st.header("Hasil Rapat", anchor=False)
        
        # Masukkan hasil ke dalam Card juga
        with st.container(border=True):
            tab_sum, tab_trans = st.tabs(["Ringkasan", "Transkripsi Lengkap"])

            with tab_sum:
                st.text_area("Ringkasan Rapat", st.session_state["summary"], height=250, label_visibility="collapsed")

            with tab_trans:
                st.text_area("Transkripsi", st.session_state["clean_transcript"], height=250, label_visibility="collapsed")

        # Tombol Download sejajar
        st.subheader("Unduh Dokumen", anchor=False)
        out_stem = Path(st.session_state["save_path"]).stem
        
        pdf_path = export_to_pdf(st.session_state["summary"], st.session_state["clean_transcript"], OUTPUT_DIR / f"{out_stem}_hasil.pdf")
        txt_path = export_to_txt(st.session_state["summary"], st.session_state["clean_transcript"], OUTPUT_DIR / f"{out_stem}_hasil.txt")

        col1, col2 = st.columns(2)
        with col1:
            with open(pdf_path, "rb") as f:
                st.download_button("Unduh Format PDF", f, file_name=pdf_path.name, mime="application/pdf", use_container_width=True, icon=":material/picture_as_pdf:")
        with col2:
            with open(txt_path, "r", encoding="utf-8") as f:
                st.download_button("Unduh Format TXT", f.read(), file_name=txt_path.name, mime="text/plain", use_container_width=True, icon=":material/description:")