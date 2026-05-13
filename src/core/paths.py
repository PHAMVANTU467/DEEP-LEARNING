import os
from pathlib import Path

# Thư mục gốc của project (nơi chứa thư mục src)
PROJECT_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CORRUPTED_DATA_DIR = DATA_DIR / "corrupted_images"

CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
HEATMAPS_DIR = PROJECT_ROOT / "heatmaps"

def ensure_dirs():
    """Tự động tạo các thư mục cần thiết nếu chưa tồn tại."""
    for d in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, CORRUPTED_DATA_DIR, CHECKPOINTS_DIR, LOGS_DIR, HEATMAPS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

ensure_dirs()

def get_checkpoint_dir(dataset_name: str, arch: str) -> Path:
    d = CHECKPOINTS_DIR / dataset_name / arch
    d.mkdir(parents=True, exist_ok=True)
    return d

def get_best_model_path(dataset_name: str, arch: str) -> Path:
    return get_checkpoint_dir(dataset_name, arch) / "best_model.pth"

def get_history_path(dataset_name: str, arch: str) -> Path:
    return get_checkpoint_dir(dataset_name, arch) / "history.json"
