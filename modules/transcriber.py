"""
modules/transcriber.py
======================
Modul transkripsi audio menggunakan model Whisper (lokal / offline).

Mendukung input:
  - WAV  (langsung diproses)
  - MP3  (langsung diproses)
  - MP4  (di ekstrak ke WAV dulu via audio_utils.py)
"""

from pathlib import Path
from typing import Callable, Optional

import whisper


# Model Whisper : tiny | base | small | medium | large-v3
WHISPER_MODEL    = "medium"
WHISPER_LANGUAGE = "id"   # Bahasa Indonesia

_loaded_models: dict = {}


def _get_model(model_name: str) -> whisper.Whisper:
    if model_name not in _loaded_models:
        import time
        from modules.performance import log_performance, get_ram_usage_mb, get_model_size_mb

        t0 = time.time()
        model = whisper.load_model(model_name)
        elapsed = time.time() - t0
        
        _loaded_models[model_name] = model
        size_mb = get_model_size_mb(model)
        log_performance(f"[COLD START] Loaded Whisper '{model_name}' in {elapsed:.2f}s | Size: {size_mb:.2f} MB | System RAM: {get_ram_usage_mb():.2f} MB")
        
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
