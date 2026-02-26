# agent.md — Instruksi AI Assistant (Cursor / Copilot / Antigravity)

## Konteks Proyek

Ini adalah proyek **Skripsi**: _Sistem Ringkasan Rapat Otomatis Berbahasa Indonesia_
menggunakan pipeline **Whisper → IndoT5 Fine-tuned** dengan antarmuka **Streamlit**.
**Fokus Pengujian**: Membandingkan performa (waktu inferensi dan kualitas) antara input format Video (MP4) dan Audio (MP3/WAV).

## Batasan Sistem (System Constraints)

- **Hardware**: RAM 8GB dengan GPU lokal. Optimasi memori adalah prioritas MUTLAK.
- **Limitasi Input**: Sistem maksimal menerima file berdurasi 40 menit pada fase aplikasi web.
- **Eksekusi**: DILARANG KERAS menggunakan API eksternal berbayar (OpenAI API, dll). Semua model berjalan **lokal/offline**.

## Stack Teknologi

- **Bahasa**: Python 3.10+
- **ASR**: `openai-whisper` (model `medium` atau `large-v3`)
- **Summarizer**: `IndoT5` fine-tuned, disimpan di `models/indot5_finetuned/`
- **UI**: Streamlit (`app.py`)
- **Input**: MP4 (ekstrak audio dulu ke 16kHz), MP3, WAV
- **Output**: PDF / TXT hasil ringkasan

## Aturan Coding

1. Semua fungsi wajib punya **docstring Bahasa Indonesia** dan **type hints**.
2. Tangani error dengan `try/except` yang informatif — jangan biarkan web crash tanpa pesan.
3. Tulis log ke `st.write` atau `st.spinner` agar penguji bisa melihat progress komputasi.
4. **TEKNIK CHUNKING (WAJIB)**: Jangan pernah menyuapkan teks panjang utuh ke IndoT5. Pecah teks berdasarkan batas token maksimal, proses per chunk, lalu gabungkan (concatenate) hasilnya.
5. **OPTIMASI MEMORI**: Selalu hapus variabel berukuran besar (`del variabel`) dan panggil `torch.cuda.empty_cache()` setelah inferensi selesai untuk mencegah memory leak.
6. Jangan hardcode path; gunakan `pathlib.Path` dan variabel konfigurasi.

## Struktur Modul (modules/)

| File             | Tanggung Jawab                                                  |
| ---------------- | --------------------------------------------------------------- |
| `audio_utils.py` | Ekstrak audio dari MP4, konversi format audio, resampling 16kHz |
| `transcriber.py` | Load & jalankan Whisper, kembalikan teks transkripsi            |
| `filter_kata.py` | Hapus kata kasar, bersihkan teks dengan regex, hapus stop words |
| `summarizer.py`  | Load IndoT5, jalankan proses chunking & summarization           |
| `exporter.py`    | Buat file PDF/TXT dari hasil ringkasan                          |

## Dataset & Eksperimen

- Data mentah ada di `dataset/01_raw/` (dibagi ke sub-folder `video_mp4/`, `audio_mp3/`, `audio_wav/`).
- Transkripsi hasil Whisper disimpan ke `dataset/02_extracted/whisper_transcripts/`.
- Data berpasangan (input-output) ada di `dataset/03_paired/` dalam format CSV dengan kolom: `source` (transkripsi) dan `target` (ringkasan ground truth).
- Total data 30: Dipisah menjadi `train.csv` (24 data) dan `test.csv` (6 data berdurasi terpendek).

## Yang JANGAN Dilakukan

- Jangan push file model (`.bin`, `.pt`, `safetensors`) ke Git.
- Jangan push file media (`.mp4`, `.mp3`, `.wav`, `.pdf`) ke Git.
