"""
modules/transcriber.py
======================
Modul transkripsi audio menggunakan model Whisper (lokal / offline).

Mendukung input:
  - WAV  (langsung diproses)
  - MP3  (langsung diproses)
  - MP4  (sebaiknya ekstrak ke WAV dulu via audio_utils.py)
"""

from pathlib import Path
from typing import Callable, Optional

import whisper


# ── Konfigurasi sistem (tidak diubah oleh pengguna) ──────────
# Model Whisper yang digunakan: tiny | base | small | medium | large-v3
# 'small' direkomendasikan untuk keseimbangan kecepatan & akurasi di CPU.
# Ganti ke 'medium' jika menggunakan GPU.
WHISPER_MODEL    = "medium"
WHISPER_LANGUAGE = "id"   # Bahasa Indonesia

# ── Cache model ───────────────────────────────────────────────

_loaded_models: dict = {}


def _get_model(model_name: str) -> whisper.Whisper:
    """
    Load model Whisper dari cache atau dari disk jika belum di-load.

    Parameter:
        model_name : Nama model Whisper ('tiny','base','small','medium','large-v3').

    Return:
        Objek model Whisper yang sudah siap digunakan.
    """
    if model_name not in _loaded_models:
        _loaded_models[model_name] = whisper.load_model(model_name)
    return _loaded_models[model_name]


def transcribe_audio(
    audio_path: Path,
    model_name: str = WHISPER_MODEL,
    language: str = WHISPER_LANGUAGE,
    task: str = "transcribe",
    progress_callback: Optional[Callable[[float], None]] = None,
) -> str:
    """
    Transkripsi file audio menggunakan Whisper.

    Parameter:
        audio_path        : Path ke file audio (WAV / MP3).
        model_name        : Nama model Whisper yang digunakan. Default 'medium'.
        language          : Kode bahasa. Default 'id' (Bahasa Indonesia).
        task              : 'transcribe' atau 'translate'.
        progress_callback : Opsional. Fungsi callable(float) yang dipanggil
                            tiap tick progress dengan nilai 0.0–1.0.
                            Digunakan untuk mengupdate st.progress() di Streamlit.

    Return:
        String teks hasil transkripsi.

    Raises:
        FileNotFoundError : Jika file audio tidak ditemukan.
        RuntimeError      : Jika proses transkripsi gagal.
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

    try:
        model = _get_model(model_name)

        # ── Intercept tqdm Whisper untuk progress bar Streamlit ──
        if progress_callback is not None:
            import tqdm as _tqdm_module
            from tqdm import tqdm as _orig_tqdm

            class _StreamlitTqdm(_orig_tqdm):
                def update(self, n: int = 1) -> None:
                    super().update(n)
                    if self.total and self.total > 0:
                        progress_callback(min(self.n / self.total, 1.0))

            _tqdm_module.tqdm = _StreamlitTqdm

            try:
                result = model.transcribe(
                    str(audio_path),
                    language=language,
                    task=task,
                    verbose=False,
                )
            finally:
                _tqdm_module.tqdm = _orig_tqdm
        else:
            result = model.transcribe(
                str(audio_path),
                language=language,
                task=task,
                verbose=False,
            )

        return result["text"].strip()

    except Exception as exc:
        raise RuntimeError(
            f"Transkripsi gagal untuk file: {audio_path}\n"
            f"Detail error: {exc}"
        ) from exc


def transcribe_with_timestamps(
    audio_path: Path,
    model_name: str = "medium",
    language: str = "id",
) -> list[dict]:
    """
    Transkripsi audio dengan informasi timestamp per segmen.

    Return:
        List of dict dengan key: 'start', 'end', 'text'.
        Contoh: [{'start': 0.0, 'end': 3.5, 'text': 'Rapat dimulai...'}, ...]
    """
    audio_path = Path(audio_path)

    if not audio_path.exists():
        raise FileNotFoundError(f"File audio tidak ditemukan: {audio_path}")

    model = _get_model(model_name)
    result = model.transcribe(str(audio_path), language=language, verbose=False)

    segments = [
        {
            "start": seg["start"],
            "end": seg["end"],
            "text": seg["text"].strip(),
        }
        for seg in result.get("segments", [])
    ]
    return segments
