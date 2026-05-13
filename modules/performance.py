import time
import psutil
import os
from pathlib import Path
from datetime import datetime
import torch

LOG_DIR = Path("data/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
PERFORMANCE_LOG = LOG_DIR / "system_performance.log"

def log_performance(message: str):
    """Menulis pesan ke file log performa."""
    with open(PERFORMANCE_LOG, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        f.write(f"[{timestamp}] {message}\n")

def get_ram_usage_mb() -> float:
    """Mengembalikan penggunaan memori (RSS) dari proses saat ini (Streamlit) dalam MegaBytes."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)

def get_model_size_mb(model: torch.nn.Module) -> float:
    """Mengembalikan ukuran model PyTorch di memori berdasarkan jumlah parameternya dalam MegaBytes."""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    return (param_size + buffer_size) / (1024 ** 2)
