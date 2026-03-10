# 🎙️ Meeting Summarizer

> **Sistem Transkripsi & Ringkasan Rapat Otomatis Berbahasa Indonesia — Offline**

Proyek skripsi ini membangun aplikasi web yang mampu mengubah rekaman rapat (audio/video) menjadi **transkripsi teks** dan **ringkasan otomatis** dalam Bahasa Indonesia. Seluruh proses berjalan secara **offline** tanpa memerlukan koneksi internet atau layanan cloud pihak ketiga.

---

## 📋 Daftar Isi

- [Fitur Utama](#-fitur-utama)
- [Arsitektur Sistem](#-arsitektur-sistem)
- [Struktur Proyek](#-struktur-proyek)
- [Tech Stack](#-tech-stack)
- [Prasyarat](#-prasyarat)
- [Instalasi](#-instalasi)
- [Tahapan Proyek](#-tahapan-proyek)
  - [Tahap 1 — Pengumpulan Data](#tahap-1--pengumpulan-data)
  - [Tahap 2 — Preprocessing & Ekstraksi Data](#tahap-2--preprocessing--ekstraksi-data)
  - [Tahap 3 — Fine-Tuning Model IndoT5](#tahap-3--fine-tuning-model-indot5)
  - [Tahap 4 — Evaluasi Model](#tahap-4--evaluasi-model)
  - [Tahap 5 — Pengembangan Aplikasi Web](#tahap-5--pengembangan-aplikasi-web)
  - [Tahap 6 — Pengujian Sistem](#tahap-6--pengujian-sistem)
- [Cara Menjalankan](#-cara-menjalankan)
- [Modul Sistem](#-modul-sistem)
- [Lisensi](#-lisensi)

---

## ✨ Fitur Utama

| Fitur                           | Deskripsi                                                                                                    |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| 🔊 **Transkripsi Otomatis**     | Mengubah audio/video rapat menjadi teks menggunakan model Whisper (OpenAI) secara lokal                      |
| 📝 **Ringkasan Otomatis**       | Meringkas transkripsi menggunakan model IndoT5 yang telah di-_fine-tune_ untuk domain rapat Bahasa Indonesia |
| 🧹 **Filter Kata Tidak Pantas** | Menyaring kata-kata kasar/tidak pantas dari hasil transkripsi secara otomatis                                |
| 📄 **Ekspor PDF & TXT**         | Mengunduh hasil transkripsi dan ringkasan dalam format PDF atau TXT                                          |
| 🎬 **Dukungan Multi-Format**    | Mendukung input file MP4 (video), MP3, dan WAV (audio)                                                       |
| 📑 **OCR Risalah Rapat**        | Mengekstrak teks dari dokumen risalah rapat PDF menggunakan Tesseract OCR                                    |
| 🔄 **Batch Transcription**      | Transkripsi otomatis banyak file audio sekaligus untuk keperluan pelatihan dataset                           |
| ⚡ **GPU Accelerated**          | Mendukung akselerasi GPU (CUDA) untuk proses transkripsi dan ringkasan yang lebih cepat                      |

---

## 🏗️ Arsitektur Sistem

```
┌──────────────────────────────────────────────────────────────┐
│                    ANTARMUKA WEB (Streamlit)                  │
│                         app.py                               │
└───────────────┬──────────────────────────────┬───────────────┘
                │                              │
                ▼                              ▼
┌───────────────────────────┐  ┌───────────────────────────────┐
│   1. EKSTRAKSI AUDIO      │  │   4. RINGKASAN TEKS           │
│   audio_utils.py          │  │   summarizer.py               │
│   MP4 → WAV (ffmpeg)      │  │   IndoT5 fine-tuned           │
└───────────┬───────────────┘  │   (chunking otomatis)         │
            │                  └───────────────┬───────────────┘
            ▼                                  │
┌───────────────────────────┐                  ▼
│   2. TRANSKRIPSI          │  ┌───────────────────────────────┐
│   transcriber.py          │  │   5. EKSPOR HASIL             │
│   Whisper (lokal)         │  │   exporter.py                 │
└───────────┬───────────────┘  │   PDF / TXT                   │
            │                  └───────────────────────────────┘
            ▼
┌───────────────────────────┐
│   3. FILTER TEKS          │
│   filter_kata.py          │
│   Hapus kata kasar +      │
│   normalisasi teks        │
└───────────────────────────┘
```

---

## 📁 Struktur Proyek

```
meeting-summarizer/
│
├── app.py                          # Aplikasi utama Streamlit (antarmuka web)
├── batch_transcribe.py             # Skrip batch transkripsi untuk dataset
├── run.ps1                         # PowerShell script untuk menjalankan aplikasi
├── requirements.txt                # Daftar dependensi Python
├── .env                            # Variabel lingkungan (API keys, konfigurasi)
├── .gitignore                      # File yang diabaikan Git
│
├── modules/                        # Modul-modul utama sistem
│   ├── __init__.py
│   ├── audio_utils.py              # Ekstraksi & konversi audio (ffmpeg)
│   ├── transcriber.py              # Transkripsi audio → teks (Whisper)
│   ├── filter_kata.py              # Filter kata kasar & normalisasi teks
│   ├── summarizer.py               # Ringkasan teks (IndoT5 fine-tuned)
│   ├── exporter.py                 # Ekspor hasil ke PDF & TXT
│   └── ocr_risalah.py              # OCR dokumen risalah rapat PDF (Tesseract)
│
├── notebooks/                      # Jupyter Notebooks untuk pipeline pelatihan
│   ├── 01_data_cleaning.ipynb      # Pembersihan & pairing data
│   ├── 02_finetuning_indot5.ipynb  # Fine-tuning model IndoT5
│   ├── 03_evaluation_rouge.ipynb   # Evaluasi model dengan skor ROUGE
│   └── 04_end_to_end_test.ipynb    # Pengujian end-to-end pipeline sistem
│
├── dataset/                        # Data untuk pelatihan model
│   ├── 01_raw/                     # Data mentah
│   │   ├── video_mp4/              # Rekaman rapat format MP4
│   │   ├── audio_mp3/              # Rekaman rapat format MP3
│   │   ├── audio_wav/              # Rekaman rapat format WAV
│   │   └── risalah_pdf/            # Dokumen risalah rapat PDF (ground truth)
│   ├── 02_extracted/               # Hasil ekstraksi (transkripsi Whisper & OCR)
│   │   └── whisper_transcripts/    # Hasil transkripsi dari batch_transcribe.py
│   ├── 03_paired/                  # Dataset yang sudah dipasangkan
│   │   ├── train.csv               # Data latih (pasangan transkripsi ↔ ringkasan)
│   │   └── test.csv                # Data uji
│   ├── data_segment_*.csv          # Data anotasi dan pembersihan teks
│   └── *_end_to_end_*.csv          # Hasil evaluasi pengujian end-to-end (MP4/MP3/WAV)
│
├── models/                         # Model machine learning
│   └── indot5_finetuned/           # Model IndoT5 hasil fine-tuning
│
└── data/                           # Data runtime aplikasi
    ├── temp_uploads/               # File upload sementara
    └── outputs/                    # Hasil output (PDF & TXT)
```

---

## 🛠️ Tech Stack

| Komponen                 | Teknologi                          | Keterangan                                                     |
| ------------------------ | ---------------------------------- | -------------------------------------------------------------- |
| **ASR (Speech-to-Text)** | OpenAI Whisper                     | Model `small` (CPU) / `medium` (GPU), bahasa Indonesia         |
| **Summarization**        | IndoT5-base                        | Model T5 Bahasa Indonesia, di-_fine-tune_ untuk domain rapat   |
| **OCR**                  | Tesseract + pdf2image              | Ekstraksi teks dari dokumen PDF risalah rapat                  |
| **Web Interface**        | Streamlit                          | Antarmuka web interaktif                                       |
| **Audio Processing**     | ffmpeg + pydub                     | Ekstraksi & konversi format audio                              |
| **Evaluasi**             | ROUGE Score                        | Metrik evaluasi kualitas ringkasan (ROUGE-1, ROUGE-2, ROUGE-L) |
| **Deep Learning**        | PyTorch + HuggingFace Transformers | Framework utama untuk model AI                                 |
| **Ekspor**               | fpdf2 + reportlab                  | Pembuatan laporan PDF                                          |

---

## 📦 Prasyarat

Sebelum memulai, pastikan perangkat kamu sudah memiliki:

1. **Python** ≥ 3.10
2. **ffmpeg** — terinstall dan tersedia di PATH sistem
   ```powershell
   # Cek instalasi ffmpeg
   ffmpeg -version
   ```
3. **Tesseract OCR** _(opsional, hanya untuk OCR risalah PDF)_
   - Download: [Tesseract di GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
4. **Poppler** _(opsional, hanya untuk OCR risalah PDF)_
   - Diperlukan oleh `pdf2image` untuk konversi PDF ke gambar
5. **GPU CUDA** _(opsional, direkomendasikan)_
   - Mempercepat proses transkripsi dan ringkasan secara signifikan

---

## 🚀 Instalasi

### 1. Clone Repository

```bash
git clone https://github.com/<username>/meeting-summarizer.git
cd meeting-summarizer
```

### 2. Buat Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install Dependensi

```powershell
pip install -r requirements.txt
```

### 4. Konfigurasi Environment

Buat file `.env` di root proyek jika diperlukan:

```env
HF_TOKEN=<token_huggingface_kamu>
```

### 5. Siapkan Model Fine-Tuned _(opsional)_

Jika sudah memiliki model IndoT5 hasil fine-tuning, letakkan file model di:

```
models/indot5_finetuned/
```

> **Catatan:** Jika model lokal tidak tersedia, sistem akan otomatis mengunduh model `Wikidepia/IndoT5-base` dari HuggingFace Hub sebagai fallback.

---

## 📐 Tahapan Proyek

Berikut adalah tahapan pengerjaan proyek dari awal hingga akhir:

### Tahap 1 — Pengumpulan Data

**Tujuan:** Mengumpulkan data rekaman rapat dan dokumen risalah sebagai ground truth.

**Langkah-langkah:**

1. Kumpulkan rekaman rapat dalam format audio (WAV/MP3) atau video (MP4)
2. Kumpulkan dokumen risalah rapat dalam format PDF sebagai referensi ringkasan
3. Simpan data mentah ke dalam folder yang sesuai:
   - `dataset/01_raw/video_mp4/` — rekaman video
   - `dataset/01_raw/audio_mp3/` — rekaman audio MP3
   - `dataset/01_raw/audio_wav/` — rekaman audio WAV
   - `dataset/01_raw/risalah_pdf/` — dokumen risalah

**Output:** Koleksi file audio/video dan dokumen PDF risalah rapat.

---

### Tahap 2 — Preprocessing & Ekstraksi Data

**Tujuan:** Mengekstrak teks dari rekaman audio (transkripsi) dan dokumen PDF (OCR) untuk membentuk dataset pelatihan.

**Langkah-langkah:**

#### 2a. Transkripsi Audio (Whisper)

Jalankan batch transcription untuk seluruh file audio:

```powershell
python batch_transcribe.py
```

Skrip ini akan:

- Memindai folder `dataset/01_raw/` untuk file audio
- Mentranskripsi setiap file menggunakan model **Whisper** (`small`)
- Menyimpan hasil transkripsi (.txt) ke `dataset/02_extracted/whisper_transcripts/`

#### 2b. Ekstraksi Teks Risalah (OCR)

Modul `ocr_risalah.py` akan:

- Mengkonversi halaman PDF ke gambar (menggunakan Poppler)
- Menjalankan OCR pada setiap halaman (menggunakan Tesseract)
- Membersihkan hasil OCR dari artefak
- Mengekstrak bagian inti pembahasan rapat untuk dijadikan referensi kosakata (vocabulary)

#### 2c. Data Cleaning & Labeling

Jalankan notebook `01_data_cleaning.ipynb` untuk:

- Membersihkan teks transkripsi dari karakter rusak dan spasi berlebih
- **OCR-guided Correction**: Menggunakan teks hasil OCR risalah sebagai _vocabulary_ lokal untuk mengoreksi otomatis _typo_ pada hasil transkripsi Whisper (menggunakan algoritma _fuzzy string matching_).
- **Pembentukan Ground Truth**: Memasangkan teks transkripsi bersih (input) dengan ringkasan referensi (target/ground truth). Ground truth ini diambil/disusun dari **dokumen risalah rapat asli** hasil ekstraksi OCR.
- Membagi dataset secara otomatis menjadi **train** (80%) dan **test** (20%) berdasar dokumen.
- Menyimpan ke `dataset/03_paired/train.csv` dan `test.csv`

**Output:** File `train.csv` dan `test.csv` berisi pasangan transkripsi ↔ ringkasan.

---

### Tahap 3 — Fine-Tuning Model IndoT5

**Tujuan:** Melatih model IndoT5-base agar mampu meringkas transkripsi rapat Bahasa Indonesia.

**Langkah-langkah:**

Jalankan notebook `02_finetuning_indot5.ipynb` (disarankan menggunakan **Google Colab** dengan GPU):

1. **Load Dataset** — Memuat `train.csv` dan `test.csv` sebagai HuggingFace Dataset
2. **Tokenisasi** — Tokenisasi pasangan input-target menggunakan tokenizer IndoT5
3. **Konfigurasi Training** — Mengatur hyperparameter:
   - Learning rate, batch size, jumlah epoch
   - Gradient checkpointing & accumulation (optimasi memori)
4. **Grid Search** — Pencarian hyperparameter terbaik secara otomatis
5. **Training** — Melatih model menggunakan HuggingFace `Trainer`
6. **Simpan Model Terbaik** — Model dengan skor ROUGE-L tertinggi disimpan ke `models/indot5_finetuned/`

**Output:** Model IndoT5 fine-tuned yang tersimpan di `models/indot5_finetuned/`.

---

### Tahap 4 — Evaluasi Model

**Tujuan:** Mengukur kualitas ringkasan yang dihasilkan model menggunakan metrik ROUGE.

**Langkah-langkah:**

Jalankan notebook `03_evaluation_rouge.ipynb`:

1. **Load Model** — Memuat model fine-tuned dari `models/indot5_finetuned/`
2. **Inferensi** — Menghasilkan ringkasan untuk seluruh data test
3. **Hitung Skor ROUGE** — Mengevaluasi kualitas ringkasan dengan metrik:
   - **ROUGE-1** — Kemiripan unigram (kata tunggal)
   - **ROUGE-2** — Kemiripan bigram (pasangan kata)
   - **ROUGE-L** — Kemiripan subsequence terpanjang
4. **Analisis Hasil** — Membandingkan ringkasan model vs ringkasan referensi

**Output:** Laporan skor ROUGE yang menunjukkan performa model.

---

### Tahap 4b — Pengujian End-to-End

**Tujuan:** Mengevaluasi kinerja sistem secara keseluruhan (waktu proses dan kualitas ringkasan) pada berbagai format input (MP4, MP3, WAV).

**Langkah-langkah:**

Jalankan notebook `04_end_to_end_test.ipynb`:

1. **Simulasi Pipeline** — Menjalankan transkripsi, filter teks, dan ringkasan secara berurutan.
2. **Uji Multi-Format** — Menguji input MP4, MP3, dan WAV untuk memastikan konsistensi output.
3. **Analisa Waktu & ROUGE** — Menghitung ROUGE score dan waktu eksekusi masing-masing format.

**Output:** Laporan dan grafik perbandingan waktu serta skor ROUGE (tersimpan di `dataset/`).

---

### Tahap 5 — Pengembangan Aplikasi Web

**Tujuan:** Membangun antarmuka web yang mudah digunakan untuk proses transkripsi dan ringkasan rapat.

**Langkah-langkah:**

1. **Pengembangan Modul** — Membangun modul-modul sistem:
   - `audio_utils.py` — Ekstraksi audio dari video (ffmpeg)
   - `transcriber.py` — Transkripsi audio menggunakan Whisper
   - `filter_kata.py` — Filter kata kasar & normalisasi teks
   - `summarizer.py` — Ringkasan teks menggunakan IndoT5 (chunking otomatis)
   - `exporter.py` — Ekspor hasil ke PDF & TXT

2. **Pengembangan Antarmuka** — Membangun `app.py` menggunakan Streamlit:
   - Upload file audio/video (MP4, MP3, WAV)
   - Tampilan progress transkripsi real-time
   - Tampilan hasil transkripsi dan ringkasan dalam tab terpisah
   - Tombol unduh PDF & TXT
   - Deteksi otomatis GPU/CPU dengan estimasi waktu

**Output:** Aplikasi web Streamlit yang berfungsi penuh.

---

### Tahap 6 — Pengujian Sistem

**Tujuan:** Memastikan seluruh pipeline berjalan dengan benar dan menghasilkan output yang sesuai.

**Langkah-langkah:**

1. **Pengujian Unit** — Menguji setiap modul secara independen
2. **Pengujian Integrasi** — Menguji alur lengkap dari upload file hingga ekspor hasil
3. **Pengujian Kualitas** — Mengevaluasi kualitas transkripsi dan ringkasan
4. **Pengujian Performa** — Mengukur waktu proses pada berbagai konfigurasi (CPU vs GPU)

---

## ▶️ Cara Menjalankan

### Opsi 1: Menggunakan Script PowerShell

```powershell
.\run.ps1
```

### Opsi 2: Menjalankan Manual

```powershell
# Aktifkan virtual environment
.\.venv\Scripts\Activate.ps1

# Jalankan aplikasi Streamlit
streamlit run app.py
```

Aplikasi akan terbuka di browser pada alamat `http://localhost:8501`.

### Alur Penggunaan Aplikasi

1. **Upload** file rekaman rapat (MP4 / MP3 / WAV)
2. Klik tombol **🚀 Mulai Proses**
3. Sistem akan menjalankan pipeline secara otomatis:
   - Ekstraksi audio (jika format MP4)
   - Transkripsi dengan Whisper
   - Filter kata tidak pantas
   - Ringkasan dengan IndoT5
4. **Lihat** hasil transkripsi dan ringkasan di tab yang tersedia
5. **Unduh** hasil dalam format PDF atau TXT

---

## 🧩 Modul Sistem

### `modules/audio_utils.py`

Utilitas pemrosesan audio menggunakan **ffmpeg**:

- `extract_audio_from_video()` — Ekstrak track audio dari video MP4 ke WAV (16kHz, mono)
- `convert_audio()` — Konversi format audio antar format

### `modules/transcriber.py`

Modul transkripsi menggunakan **Whisper**:

- `transcribe_audio()` — Transkripsi file audio ke teks dengan progress callback
- `transcribe_with_timestamps()` — Transkripsi dengan informasi timestamp per segmen
- Cache model untuk menghindari loading ulang

### `modules/filter_kata.py`

Modul pembersihan teks hasil transkripsi:

- `filter_kata_kasar()` — Menyensor kata kasar menggunakan regex
- `normalisasi_teks()` — Normalisasi spasi, karakter kontrol, dan baris kosong
- `filter_teks()` — Pipeline pembersihan lengkap
- `muat_kata_kasar_dari_file()` — Memuat daftar kata kasar dari file eksternal

### `modules/summarizer.py`

Modul ringkasan teks menggunakan **IndoT5**:

- Chunking otomatis: teks dipecah menjadi potongan ≤300 kata
- Setiap chunk diringkas secara independen menggunakan beam search
- Hasil digabungkan; jika masih terlalu panjang, dilakukan ringkasan bertingkat
- Fallback ke model HuggingFace Hub jika model lokal tidak tersedia

### `modules/exporter.py`

Modul ekspor hasil ke file:

- `export_to_pdf()` — Membuat laporan PDF dengan header, ringkasan, dan transkripsi lengkap
- `export_to_txt()` — Membuat laporan plain text

### `modules/ocr_risalah.py`

Modul OCR untuk dokumen risalah rapat PDF:

- `pdf_ke_gambar()` — Konversi PDF ke gambar per halaman
- `ocr_gambar()` — OCR menggunakan Tesseract
- `bersihkan_teks_ocr()` — Pembersihan artefak OCR
- `ekstrak_bagian()` — Ekstraksi bagian inti pembahasan
- `proses_pdf_risalah()` — Pipeline OCR lengkap (PDF → teks bersih)
- `proses_semua_pdf()` — Batch processing seluruh PDF dalam direktori

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan **Skripsi**. Silakan hubungi pembuat untuk informasi lebih lanjut mengenai penggunaan.
