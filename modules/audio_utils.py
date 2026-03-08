"""
modules/audio_utils.py
======================
Utilitas pemrosesan audio:
  - Ekstrak audio (WAV) dari file video MP4
  - Konversi format audio bila diperlukan
"""

from pathlib import Path
import subprocess


def get_media_duration(file_path: Path) -> float:
    """
    Mendapatkan durasi file media (audio/video) dalam detik menggunakan ffprobe.

    Parameter:
        file_path : Path ke file media.

    Return:
        Durasi dalam detik (float).

    Raises:
        RuntimeError : Jika ffprobe gagal membaca durasi.
    """
    file_path = Path(file_path)
    command = [
        "ffprobe",
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True,
        )
        return float(result.stdout.strip())
    except (subprocess.CalledProcessError, ValueError) as e:
        raise RuntimeError(
            f"Gagal membaca durasi file: {file_path}\n{e}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffprobe tidak ditemukan. Pastikan ffmpeg/ffprobe sudah terinstall "
            "dan tersedia di PATH sistem."
        )


def extract_audio_from_video(video_path: Path, output_dir: Path) -> Path:
    """
    Ekstrak track audio dari file video MP4 menjadi file WAV.

    Parameter:
        video_path  : Path ke file video MP4.
        output_dir  : Direktori tempat file WAV hasil ekstraksi disimpan.

    Return:
        Path ke file WAV yang baru dibuat.

    Raises:
        FileNotFoundError : Jika file video tidak ditemukan.
        RuntimeError      : Jika proses ffmpeg gagal.
    """
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not video_path.exists():
        raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

    audio_path = output_dir / (video_path.stem + ".wav")

    command = [
        "ffmpeg",
        "-y",                   # Timpa file output jika sudah ada
        "-i", str(video_path),  # Input video
        "-vn",                  # Abaikan stream video
        "-acodec", "pcm_s16le", # Encode audio ke WAV 16-bit
        "-ar", "16000",         # Sample rate 16 kHz (optimal untuk Whisper)
        "-ac", "1",             # Mono channel
        str(audio_path),
    ]

    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(
            f"ffmpeg gagal mengekstrak audio.\n"
            f"stderr: {e.stderr}"
        ) from e
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg tidak ditemukan. Pastikan ffmpeg sudah terinstall "
            "dan tersedia di PATH sistem."
        )

    return audio_path


def convert_audio(input_path: Path, output_dir: Path, target_format: str = "wav") -> Path:
    """
    Konversi file audio ke format lain menggunakan ffmpeg.

    Parameter:
        input_path    : Path ke file audio sumber.
        output_dir    : Direktori output.
        target_format : Format target, default 'wav'.

    Return:
        Path ke file audio hasil konversi.
    """
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / f"{input_path.stem}.{target_format}"

    command = [
        "ffmpeg",
        "-y",
        "-i", str(input_path),
        "-ar", "16000",
        "-ac", "1",
        str(output_path),
    ]

    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Konversi audio gagal.\nstderr: {e.stderr}") from e

    return output_path
