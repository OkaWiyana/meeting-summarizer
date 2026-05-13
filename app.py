# Antarmuka utama aplikasi menggunakan streamlit
# ======================================================
# cara run manual: 
# 1. buka terminal
# 2. aktifkan virtual environment: .\venv\Scripts\Activate.ps1
# 3. jalankan perintah: streamlit run app.py
# ======================================================

# Catat waktu mulai server untuk menghitung total cold start
import time as _time
_SERVER_START_TIME = _time.time()

import streamlit as st
import torch
import shutil
from pathlib import Path

from modules.audio_utils import extract_audio_from_video, get_media_duration
from modules.transcriber import transcribe_audio, WHISPER_MODEL
from modules.filter_kata import filter_teks
from modules.summarizer import summarize_text
from modules.exporter import export_to_pdf, export_to_txt

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="Peringkas Rapat Otomatis",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Styling CSS Custom
st.markdown("""
<style>
    /* Mengatur jarak halaman */
    .block-container {
        padding-top: 2rem !important;
        max-width: 800px !important;
    }
    
    /* Desain Tombol Utama */
    .stButton>button[kind="primary"] {
        border-radius: 8px !important;
        font-weight: bold !important;
        padding: 0.5rem 1rem !important;
    }
    
    /* Radius Text Area */
    .stTextArea textarea {
        border-radius: 10px !important;
        line-height: 1.6 !important;
    }
    
    /* Mengubah teks tombol 'Browse files' menjadi 'Upload File' */
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"] {
        text-indent: -9999px;
        line-height: 0;
    }
    
    [data-testid="stFileUploader"] [data-testid="stBaseButton-secondary"]::after {
        content: "Upload File";
        line-height: initial;
        text-indent: 0;
        display: block;
    }
</style>
""", unsafe_allow_html=True)

# Pengaturan Folder Temp & Output
UPLOAD_DIR = Path("data/temp_uploads")
OUTPUT_DIR = Path("data/outputs")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# extension rekaman rapat yang diizinkan
ALLOWED_EXTENSIONS = ["mp4", "mp3", "wav"]
# durasi maksimal rekaman rapat
MAX_DURATION_MINUTES = 40

# Fungsi untuk Menghapus file sementara di folder temp_uploads & outputs
def cleanup_temp_files():
    for folder in [UPLOAD_DIR, OUTPUT_DIR]:
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)
            folder.mkdir(parents=True, exist_ok=True)


# Inisialisasi variabel session Streamlit
for key in ["raw_transcript", "clean_transcript", "summary",
            "save_path", "processed", "audio_path", "inference_latency"]:
    if key not in st.session_state:
        st.session_state[key] = None

# Reset cache saat buka halaman baru
if "_session_initialized" not in st.session_state:
    cleanup_temp_files()
    st.session_state["_session_initialized"] = True

# Proses Pre-Load Model (Pindah Cold Start ke Awal)
if "_models_loaded" not in st.session_state:
    with st.spinner("Sistem sedang memuat memori Model AI, mohon tunggu..."):
        import time
        from modules.performance import log_performance
        
        start_preload = time.time()
        
        from modules.transcriber import _get_model, WHISPER_MODEL
        from modules.summarizer import _load_model, MODEL_LOCAL_PATH
        
        _get_model(WHISPER_MODEL)
        _load_model(MODEL_LOCAL_PATH)
        
        total_cold_start = time.time() - start_preload
        log_performance(f"--- WAKTU LOAD MODEL (Whisper + IndoT5): {total_cold_start:.2f} detik ---")
        
        # Hitung total waktu dari server start hingga sistem siap
        total_server_to_ready = time.time() - _SERVER_START_TIME
        log_performance(f"--- TOTAL COLD START (Server Start → Siap Digunakan): {total_server_to_ready:.2f} detik ---")
        log_performance(f"    STATUS: {'LULUS' if total_server_to_ready <= 30 else 'GAGAL'} (Threshold: <= 30 detik)")
        
        st.session_state["_cold_start_time"] = round(total_server_to_ready, 2)
        
    st.session_state["_models_loaded"] = True

# Section bagian Sidebar Info Sistem, kiri bawah
with st.sidebar:
    st.title("Sistem")
    st.caption("Informasi teknis mesin pemroses.")
    
    # menampilkan informasi sistem berjalan pada GPU atau CPU
    _device = "GPU (CUDA)" if torch.cuda.is_available() else "CPU"
    if torch.cuda.is_available():
        st.success(f"Akselerasi: **{_device}**", icon=":material/speed:")
    else:
        st.warning(f"Berjalan di: **{_device}**\n\nProses mungkin memakan waktu.", icon=":material/memory:")
        
    # menampilkan spesifikasi model yang digunakan
    st.divider()
    st.markdown("**Spesifikasi Model:**")
    st.markdown(f"- **ASR:** Whisper `{WHISPER_MODEL}`\n- **NLP:** IndoT5\n- **Chunking:** Otomatis")
    
    # st.divider()
    # st.caption("**Tip Mode Gelap:**\nKlik menu ⋮ di kanan atas > Settings > Theme > Pilih Dark/Light.")

# Section bagian Halaman Utama Aplikasi, center
st.title("Peringkas Rapat Otomatis")
st.markdown("Unggah rekaman rapat, dan sistem akan membuat transkripsi serta ringkasan secara otomatis.")

# Section bagian Upload File
uploaded_file = st.file_uploader(
    "Pilih file rekaman (MP4, MP3, WAV)",
    label_visibility="collapsed"
)

# Section bagian Preview File & Aksi Button
if uploaded_file is not None:
    file_ext = uploaded_file.name.rsplit(".", 1)[-1].lower()
    
    if file_ext not in ALLOWED_EXTENSIONS:
        st.error("Format file tidak didukung", icon=":material/error:")
        st.stop()

    save_path = UPLOAD_DIR / uploaded_file.name

    if st.session_state["save_path"] != str(save_path):
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.session_state["save_path"] = str(save_path)
        st.session_state["processed"] = False
        st.session_state["summary"] = None

    # Cek durasi maksimal audio
    try:
        duration_seconds = get_media_duration(save_path)
        duration_minutes = duration_seconds / 60
        st.session_state["media_duration"] = duration_seconds
    except RuntimeError:
        duration_seconds = None
        duration_minutes = None
        st.session_state["media_duration"] = None

    duration_ok = (duration_minutes is not None and duration_minutes <= MAX_DURATION_MINUTES)

    if duration_minutes is not None and duration_minutes > MAX_DURATION_MINUTES:
        st.error(
            f"Durasi file terlalu panjang: **{int(duration_minutes)} menit {int(duration_seconds % 60)} detik**.\n\n"
            f"Batas maksimal adalah **{MAX_DURATION_MINUTES} menit**. "
            f"Silakan unggah rekaman yang lebih pendek.",
            icon=":material/timer_off:"
        )
        st.stop()

    # SectionPreview File & Aksi Button
    with st.container(border=True):
        st.subheader("Detail File", anchor=False)
        st.markdown(f"**Nama file:** `{uploaded_file.name}`")

        if duration_minutes is not None:
            st.markdown(f"**Durasi:** {int(duration_minutes)} menit {int(duration_seconds % 60)} detik")
        
        # Audio/Video preview
        if file_ext == "mp4":
            st.video(str(save_path))
        else:
            st.audio(str(save_path))

        # Tombol Eksekusi
        run_btn = st.button("Mulai Proses", use_container_width=True, type="primary")

    # Section: Proses Utama AI
    if run_btn:
        from modules.performance import log_performance, get_ram_usage_mb
        import time
        
        start_time_inference = time.time()
        log_performance(f"--- Mulai Inferensi pada {uploaded_file.name} ---")
        log_performance(f"RAM awal sebelum proses: {get_ram_usage_mb():.2f} MB")

        st.session_state["processed"] = False
        audio_path = save_path

        # Mencegah reload browser tidak disengaja
        import streamlit.components.v1 as components
        components.html("""
            <script>
                window.parent.addEventListener('beforeunload', function(e) {
                    e.preventDefault();
                    e.returnValue = '';
                });
            </script>
        """, height=0)

        # Status Loading dropdown
        with st.status("Memproses rapat, mohon tunggu...", expanded=True) as status_box:
            try:
                if save_path.suffix.lower() == ".mp4":
                    st.write("Mengekstrak audio dari video...")
                    audio_path = extract_audio_from_video(save_path, UPLOAD_DIR)
                st.session_state["audio_path"] = str(audio_path)

                st.write("Mentranskripsi percakapan...")
                
                # Progress bar untuk transkripsi
                _pbar = st.progress(0, text="Persiapan transkripsi: 0%")
                
                def _update_pbar(pct: float):
                    # Fungsi update nilai progress bar
                    safe_pct = max(0.0, min(pct, 1.0))
                    _pbar.progress(safe_pct, text=f"Proses transkripsi: {int(safe_pct * 100)}%")

                raw_transcript = transcribe_audio(audio_path, progress_callback=_update_pbar)
                
                # Tandai progress bar mencapai 100%
                _pbar.progress(1.0, text="Transkripsi selesai: 100%")
                st.session_state["raw_transcript"] = raw_transcript

                st.write("Membersihkan teks...")
                clean_transcript = filter_teks(raw_transcript)
                st.session_state["clean_transcript"] = clean_transcript

                st.write("Menyusun ringkasan utama...")
                _pbar_sum = st.progress(0, text="Persiapan ringkasan: 0%")
                
                def _update_pbar_sum(pct: float):
                    safe_pct = max(0.0, min(pct, 1.0))
                    _pbar_sum.progress(safe_pct, text=f"Proses ringkasan: {int(safe_pct * 100)}%")

                summary = summarize_text(clean_transcript, progress_callback=_update_pbar_sum)
                _pbar_sum.progress(1.0, text="Ringkasan selesai: 100%")
                
                st.session_state["summary"] = summary
                
                st.session_state["processed"] = True

                latency = time.time() - start_time_inference
                st.session_state["inference_latency"] = latency
                log_performance(f"Latency (Waktu Inferensi Total): {latency:.2f} detik")
                
                # Hitung rasio inferensi vs durasi asli
                media_dur = st.session_state.get("media_duration")
                if media_dur and media_dur > 0:
                    rasio = latency / media_dur
                    status_rasio = "LULUS" if rasio <= 1.0 else "GAGAL"
                    log_performance(f"Durasi asli media: {media_dur:.2f} detik")
                    log_performance(f"Rasio Inferensi: {rasio:.4f}x (Target <= 1.0x) → {status_rasio}")
                
                log_performance(f"RAM akhir setelah proses: {get_ram_usage_mb():.2f} MB")
                log_performance(f"--- Inferensi Selesai ---")

                # Menutup dropdown loading
                status_box.update(label="Pemrosesan Selesai!", state="complete", expanded=False)

            except Exception as e:
                status_box.update(label="Terjadi Kesalahan", state="error", expanded=True)
                st.error("Maaf, gagal memproses data, silakan coba lagi", icon=":material/error:")
                log_performance(f"ERROR Inferensi: {str(e)}")

    # Section: Hasil & Download
    if st.session_state["processed"] and st.session_state["summary"]:
        
        st.header("Hasil Rapat", anchor=False)
        
        # Tampilkan metrik waktu proses jika ada
        if st.session_state.get("inference_latency") is not None:
            lat = st.session_state["inference_latency"]
            lat_m = int(lat // 60)
            lat_s = lat % 60
            if lat_m > 0:
                st.info(f"Waktu Proses Sistem: **{lat_m} Menit {lat_s:.1f} Detik** ({lat:.2f} detik)")
            else:
                st.info(f"Waktu Proses Sistem: **{lat_s:.2f} Detik**")
        
        # Tata Letak Hasil (Card)
        with st.container(border=True):
            tab_sum, tab_trans = st.tabs(["Ringkasan", "Transkripsi Lengkap"])

            with tab_sum:
                st.text_area("Ringkasan Rapat", st.session_state["summary"], height=250, label_visibility="collapsed")

            with tab_trans:
                st.text_area("Transkripsi", st.session_state["clean_transcript"], height=250, label_visibility="collapsed")

        # Tata Letak Tombol Download
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