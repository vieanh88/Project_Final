"""
=============================================================================
  BƯỚC 1: DOWNLOAD — Tải dataset ViVoice từ Hugging Face
=============================================================================
Mục tiêu: Tải toàn bộ dataset capleaf/viVoice (file .parquet) về ổ cứng
          local để các bước tiếp theo xử lý offline.

Chạy lệnh:
    python step1_download.py
    python step1_download.py --config custom_config.yaml
=============================================================================
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml
from dotenv import load_dotenv

# =============================================================================
# KHẮC PHỤC LỖI ENCODING TRÊN WINDOWS
# =============================================================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class DownloadConfig:
    """Cấu hình cho bước tải dataset từ Hugging Face."""

    # Thông tin dataset
    dataset_name: str = "capleaf/viVoice"
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None

    # Đường dẫn
    work_dir: str = "./workdir"
    cache_subdir: str = "hf_cache"

    # Tùy chọn
    num_proc: int = 4
    skip_existing: bool = True

    # HuggingFace token (đọc từ .env nếu cần dataset private)
    hf_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DownloadConfig":
        """Load config từ file YAML chung của prepare_vivoice."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step1 = full_config.get("step1_download", {})

        return cls(
            dataset_name=step1.get("dataset_name", cls.dataset_name),
            dataset_config=step1.get("dataset_config", cls.dataset_config),
            dataset_split=step1.get("dataset_split", cls.dataset_split),
            work_dir=paths.get("work_dir", cls.work_dir),
            cache_subdir=step1.get("cache_subdir", cls.cache_subdir),
            num_proc=step1.get("num_proc", cls.num_proc),
            skip_existing=step1.get("skip_existing", cls.skip_existing),
        )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step1_download.log"

    # Xóa handler cũ nếu có (tránh duplicate khi chạy lại)
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("step1_download")


# =============================================================================
# CORE LOGIC
# =============================================================================

def check_cache_exists(cache_dir: Path, dataset_name: str) -> bool:
    """
    Kiểm tra xem dataset đã được tải về cache chưa.
    Heuristic: Nếu thư mục cache tồn tại và chứa ít nhất 1 file .arrow
    hoặc .parquet, coi như đã tải xong.
    """
    if not cache_dir.exists():
        return False

    # Tìm kiếm đệ quy các file dữ liệu trong cache
    data_files = (
        list(cache_dir.rglob("*.arrow"))
        + list(cache_dir.rglob("*.parquet"))
    )
    return len(data_files) > 0


def download_dataset(config: DownloadConfig, logger: logging.Logger) -> Path:
    """
    Tải dataset từ Hugging Face Hub về local cache.

    Returns:
        Path tới thư mục cache chứa dataset đã tải.
    """
    # --- Lazy import (chỉ import khi thực sự cần) ---
    from datasets import load_dataset

    cache_dir = Path(config.work_dir) / config.cache_subdir
    cache_dir.mkdir(parents=True, exist_ok=True)

    # --- Kiểm tra cache ---
    if config.skip_existing and check_cache_exists(cache_dir, config.dataset_name):
        logger.info(f"Cache đã tồn tại tại: {cache_dir}")
        logger.info("Bỏ qua bước tải (skip_existing=true). Xóa thư mục cache nếu muốn tải lại.")
        return cache_dir

    # --- Log thông tin ---
    logger.info(f"Dataset       : {config.dataset_name}")
    logger.info(f"Config/Subset : {config.dataset_config or '(mặc định)'}")
    logger.info(f"Split         : {config.dataset_split or '(tất cả)'}")
    logger.info(f"Cache dir     : {cache_dir}")
    logger.info(f"Num proc      : {config.num_proc}")

    # --- Chuẩn bị tham số ---
    load_kwargs = {
        "path": config.dataset_name,
        "cache_dir": str(cache_dir),
        "num_proc": config.num_proc
    }

    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config

    if config.dataset_split:
        load_kwargs["split"] = config.dataset_split

    if config.hf_token:
        load_kwargs["token"] = config.hf_token

    # --- Bắt đầu tải ---
    logger.info("=" * 60)
    logger.info("Bắt đầu tải dataset từ Hugging Face Hub...")
    logger.info("(Lần đầu có thể mất rất lâu tùy kích thước dataset và tốc độ mạng)")
    logger.info("=" * 60)

    start_time = time.time()

    try:
        dataset = load_dataset(**load_kwargs)
    except Exception as e:
        logger.error(f"Lỗi khi tải dataset: {e}")
        logger.error("Kiểm tra lại: (1) Tên dataset, (2) Kết nối mạng, (3) HF token nếu dataset private.")
        raise

    elapsed = time.time() - start_time

    # --- Thống kê kết quả ---
    logger.info("=" * 60)
    logger.info("TẢI DATASET HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"Thời gian      : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")

    # In thông tin từng split
    if hasattr(dataset, "keys"):
        # DatasetDict (nhiều splits)
        for split_name, split_data in dataset.items():
            logger.info(f"  Split '{split_name}': {len(split_data):,} samples")
            if len(split_data) > 0:
                sample = split_data[0]
                logger.info(f"    Columns    : {list(sample.keys())}")
                if "text" in sample:
                    text_preview = sample["text"][:80] + "..." if len(sample["text"]) > 80 else sample["text"]
                    logger.info(f"    Text mẫu   : {text_preview}")
                if "audio" in sample:
                    audio_info = sample["audio"]
                    if isinstance(audio_info, dict):
                        sr = audio_info.get("sampling_rate", "N/A")
                        arr = audio_info.get("array", None)
                        dur = f"{len(arr) / sr:.2f}s" if arr is not None and sr != "N/A" else "N/A"
                        logger.info(f"    Audio mẫu  : sr={sr}, duration={dur}")
    else:
        # Dataset đơn (1 split)
        logger.info(f"  Tổng samples : {len(dataset):,}")

    logger.info(f"  Cache tại    : {cache_dir}")

    return cache_dir

# MAIN
def main():
    # --- Parse arguments ---
    parser = argparse.ArgumentParser(
        description="Bước 1: Tải dataset ViVoice từ Hugging Face Hub"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml (mặc định: config.yaml trong cùng thư mục)",
    )
    args = parser.parse_args()

    # --- Load .env (nếu có) để lấy HF_TOKEN ---
    env_path = Path(args.config).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        print(f"  Hãy đảm bảo file config.yaml nằm cùng thư mục hoặc chỉ định đường dẫn bằng --config")
        sys.exit(1)

    config = DownloadConfig.from_yaml(str(config_path))

    # Ghi đè HF token từ biến môi trường (nếu có)
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if hf_token:
        config.hf_token = hf_token

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  BƯỚC 1: TẢI DATASET VIVOICE TỪ HUGGING FACE")
    logger.info("=" * 60)
    logger.info(f"Config file    : {config_path.resolve()}")
    logger.info(f"HF Token       : {'Có (từ .env)' if config.hf_token else 'Không (public dataset)'}")

    # --- Chạy ---
    try:
        cache_dir = download_dataset(config, logger)
        logger.info("")
        logger.info("Bước tiếp theo: Chạy step2_extract_audio.py để giải mã audio → .wav")
        logger.info("=" * 60)
    except Exception as e:
        logger.error(f"THẤT BẠI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()