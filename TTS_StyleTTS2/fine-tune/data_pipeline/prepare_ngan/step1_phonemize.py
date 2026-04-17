"""
=============================================================================
  PREPARE_NGAN — BƯỚC 1: PHONEMIZE
=============================================================================
Mục tiêu: Đọc filelist giọng Bác Ngạn (đã qua pipeline 7 bước: Demucs,
          DeepFilterNet, Pyannote, Whisper, DNSMOS), làm sạch text thô
          (xóa số, ký tự đặc biệt), rồi chuyển sang chuỗi phoneme IPA.

Đầu vào : filelist_train.txt & filelist_val.txt từ output_dataset/
           Format LJSpeech: wav_path|raw_text

Đầu ra  : workdir/ngan_train_phoneme.txt  (wav_path|phoneme)
           workdir/ngan_val_phoneme.txt    (wav_path|phoneme)

Chạy lệnh:
    python step1_phonemize.py
    python step1_phonemize.py --config config.yaml
    python step1_phonemize.py --max-lines 50   (test nhanh)
=============================================================================
"""

import os
import sys
import re
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass

import yaml

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
# KHẮC PHỤC LỖI "WinError 193" — Bypass Linux Binary trong vinorm
# (Copy nguyên khối logic đã chạy hoàn hảo từ step08_phonemize.py)
# =============================================================================
import vinorm

def _mock_tts_norm(text, *args, **kwargs):
    """Hàm giả thay thế cho TTSnorm — bỏ qua binary Linux, chỉ lowercase."""
    return str(text).lower().strip()

vinorm.TTSnorm = _mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

import viphoneme
viphoneme.TTSnorm = _mock_tts_norm

from viphoneme import vi2IPA_split

# CONFIGURATION
# Bảng chuyển số → chữ tiếng Việt (0-9)
DIGIT_TO_WORD = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
}

@dataclass
class NganPhonemizeConfig:
    """Cấu hình cho bước phonemize dataset Bác Ngạn."""

    # Đường dẫn tới thư mục output_dataset của pipeline 7 bước
    ngan_dataset_dir: str = ""

    # Tên file filelist (format LJSpeech: wav_path|text)
    input_filelists: list = None

    # Thư mục work (lưu file trung gian, log)
    work_dir: str = "./workdir"

    # Có chuyển số thành chữ không (true = "123" → "một hai ba")
    convert_numbers_to_words: bool = True

    # Có xóa hoàn toàn số không (chỉ áp dụng nếu convert_numbers = false)
    remove_numbers: bool = False

    # Số dòng tối đa (0 = tất cả)
    max_lines: int = 0

    # Bỏ qua nếu output đã tồn tại
    skip_existing: bool = True

    def __post_init__(self):
        if self.input_filelists is None:
            self.input_filelists = ["filelist_train.txt", "filelist_val.txt"]

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NganPhonemizeConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        ngan = full_config.get("prepare_ngan", {})
        paths = full_config.get("paths", {})

        config = cls(
            ngan_dataset_dir=ngan.get("dataset_dir", cls.ngan_dataset_dir),
            input_filelists=ngan.get("input_filelists", None),
            work_dir=paths.get("work_dir", ngan.get("work_dir", cls.work_dir)),
            convert_numbers_to_words=ngan.get("convert_numbers_to_words", cls.convert_numbers_to_words),
            remove_numbers=ngan.get("remove_numbers", cls.remove_numbers),
            skip_existing=ngan.get("skip_existing", cls.skip_existing),
        )
        return config

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ngan_step1_phonemize.log"

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
    return logging.getLogger("ngan_step1")

# TEXT CLEANING — Đặc biệt cho dataset Bác Ngạn
def digits_to_words(text: str) -> str:
    """
    Chuyển từng chữ số đơn lẻ thành chữ tiếng Việt.
    Ví dụ: "phòng 304" → "phòng ba không bốn"
    """
    result = []
    for char in text:
        if char.isdigit():
            result.append(DIGIT_TO_WORD.get(char, char))
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)

def clean_ngan_text(text: str, convert_numbers: bool = True, remove_numbers: bool = False) -> str:
    """
    Làm sạch text từ transcript Whisper của Bác Ngạn:
    1. Loại bỏ ngoặc kép, ngoặc đơn, dấu ngoặc vuông
    2. Xử lý số (chuyển thành chữ hoặc xóa)
    3. Chỉ giữ lại dấu câu cơ bản: , . ? !
    4. Chuẩn hóa khoảng trắng
    """
    # Xóa các ký tự đặc biệt
    text = re.sub(r'["""\'\(\)\[\]\{\}<>«»—–_\-\+\=\*\#\@\&\^\~\`\\\/\|]', ' ', text)

    # Xóa dấu ba chấm (thường có nhiều trong truyện ma)
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    # Xử lý số
    if convert_numbers:
        text = digits_to_words(text)
    elif remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Chỉ giữ ký tự chữ (Unicode), dấu câu cơ bản, khoảng trắng
    # Giữ nguyên các ký tự tiếng Việt có dấu
    text = re.sub(r'[^\w\s,.\?!]', ' ', text)

    # Xóa underscore (từ \w match)
    text = text.replace('_', ' ')

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def text_to_phoneme(text: str) -> str:
    """
    Chuyển đổi text tiếng Việt → chuỗi IPA phoneme.
    Giống hệt logic step08_phonemize.py.
    """
    try:
        phonemes = vi2IPA_split(text, " ")
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes
    except Exception as e:
        return f"[ERROR] {str(e)}"

# CORE LOGIC
def process_filelist(
    input_path: Path,
    output_path: Path,
    config: NganPhonemizeConfig,
    logger: logging.Logger,
) -> dict:
    """
    Xử lý 1 file filelist: đọc wav_path|text → clean → phonemize → ghi wav_path|phoneme.

    Returns:
        dict thống kê kết quả
    """
    from tqdm import tqdm

    stats = {
        "total": 0,
        "success": 0,
        "error_g2p": 0,
        "empty_text": 0,
        "empty_phoneme": 0,
    }

    if not input_path.exists():
        logger.warning(f"Bỏ qua (không tìm thấy): {input_path}")
        return stats

    # Kiểm tra skip
    if config.skip_existing and output_path.exists():
        line_count = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info(f"Đã tồn tại: {output_path.name} ({line_count:,} dòng) → bỏ qua")
        return stats

    # Đọc file
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đã tải {len(lines):,} dòng từ {input_path.name}")

    # Giới hạn nếu test
    if config.max_lines > 0:
        lines = lines[: config.max_lines]
        logger.info(f"Giới hạn: {config.max_lines} dòng (chế độ test)")

    processed_records = []
    error_records = []

    for line in tqdm(lines, desc=f"Phonemize {input_path.name}", ncols=100):
        line = line.strip()
        if not line:
            continue

        stats["total"] += 1
        parts = line.split("|")

        # Format: wav_path|text
        if len(parts) >= 2:
            wav_path = parts[0].strip()
            raw_text = parts[1].strip()
        elif len(parts) == 1:
            # Chỉ có text (không có wav_path)
            wav_path = ""
            raw_text = parts[0].strip()
        else:
            stats["error_g2p"] += 1
            continue

        # Bỏ text rỗng
        if not raw_text:
            stats["empty_text"] += 1
            continue

        # Clean text
        cleaned = clean_ngan_text(
            raw_text,
            convert_numbers=config.convert_numbers_to_words,
            remove_numbers=config.remove_numbers,
        )

        if not cleaned:
            stats["empty_text"] += 1
            continue

        # Phonemize
        phoneme_str = text_to_phoneme(cleaned)

        if phoneme_str.startswith("[ERROR]"):
            stats["error_g2p"] += 1
            error_records.append(f"{raw_text} → {phoneme_str}")
            if stats["error_g2p"] <= 20:
                logger.warning(f"  Lỗi G2P: {raw_text[:50]}... → {phoneme_str}")
            continue

        if not phoneme_str or phoneme_str.isspace():
            stats["empty_phoneme"] += 1
            continue

        # Ghi output
        if wav_path:
            processed_records.append(f"{wav_path}|{phoneme_str}")
        else:
            processed_records.append(phoneme_str)

        stats["success"] += 1

    # Lưu file output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_records) + "\n")

    logger.info(
        f"Hoàn tất {input_path.name}: "
        f"OK={stats['success']:,} | "
        f"Lỗi G2P={stats['error_g2p']:,} | "
        f"Text rỗng={stats['empty_text']:,}"
    )

    # Lưu error log
    if error_records:
        error_path = output_path.parent / f"{output_path.stem}_errors.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            for record in error_records:
                f.write(record + "\n")

    # In mẫu
    if processed_records:
        logger.info(f"  Mẫu kiểm tra:")
        for rec in processed_records[:3]:
            parts = rec.split("|")
            if len(parts) == 2:
                wav_name = Path(parts[0]).name
                phon = parts[1][:50] + "..." if len(parts[1]) > 50 else parts[1]
                logger.info(f"    {wav_name} | {phon}")
            else:
                logger.info(f"    {rec[:80]}")

    return stats
def main_process(config: NganPhonemizeConfig, logger: logging.Logger):
    """Quy trình chính: Phonemize tất cả filelist của Bác Ngạn."""
    work_dir = Path(config.work_dir)
    ngan_dir = Path(config.ngan_dataset_dir)

    if not ngan_dir.exists():
        logger.error(f"Thư mục dataset Bác Ngạn không tồn tại: {ngan_dir}")
        logger.error("Kiểm tra lại đường dẫn 'ngan_dataset_dir' trong config.yaml")
        return

    start_time = time.time()
    total_stats = {
        "total": 0,
        "success": 0,
        "error_g2p": 0,
        "empty_text": 0,
        "empty_phoneme": 0,
    }

    for filename in config.input_filelists:
        input_path = ngan_dir / filename
        # Output: filelist_train.txt → ngan_train_phoneme.txt
        out_name = filename.replace("filelist_", "ngan_").replace(".txt", "_phoneme.txt")
        output_path = work_dir / out_name

        logger.info("")
        logger.info(f"--- Xử lý: {filename} ---")

        stats = process_filelist(input_path, output_path, config, logger)

        for key in total_stats:
            total_stats[key] += stats[key]

    elapsed = time.time() - start_time

    # Thống kê tổng
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ PHONEMIZE DATASET BÁC NGẠN")
    logger.info("=" * 60)
    logger.info(f"  Thời gian         : {elapsed:.1f}s")
    logger.info(f"  Tổng dòng         : {total_stats['total']:,}")
    logger.info(f"  Thành công        : {total_stats['success']:,}")
    logger.info(f"  Lỗi G2P          : {total_stats['error_g2p']:,}")
    logger.info(f"  Text rỗng         : {total_stats['empty_text']:,}")
    logger.info(f"  Phoneme rỗng      : {total_stats['empty_phoneme']:,}")

    success_rate = (
        total_stats["success"] / total_stats["total"] * 100
        if total_stats["total"] > 0
        else 0
    )
    logger.info(f"  Tỷ lệ thành công : {success_rate:.1f}%")
    logger.info("")
    logger.info("  Bước tiếp theo: Chạy step2_make_filelist.py để append speaker_id & split")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Ngạn — Bước 1: Phonemize filelist Bác Ngạn"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--ngan-dir",
        type=str,
        default=None,
        help="Override đường dẫn thư mục output_dataset của Bác Ngạn",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Giới hạn số dòng xử lý (0 = tất cả)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if config_path.exists():
        config = NganPhonemizeConfig.from_yaml(str(config_path))
    else:
        logger_temp = logging.getLogger()
        logger_temp.warning(f"Config không tồn tại: {config_path}, dùng giá trị mặc định")
        config = NganPhonemizeConfig()

    # Override từ CLI
    if args.ngan_dir:
        config.ngan_dataset_dir = args.ngan_dir
    if args.max_lines > 0:
        config.max_lines = args.max_lines

    # Kiểm tra bắt buộc
    if not config.ngan_dataset_dir:
        print("[LỖI] Chưa chỉ định đường dẫn dataset Bác Ngạn!")
        print("  Dùng --ngan-dir hoặc đặt 'prepare_ngan.dataset_dir' trong config.yaml")
        print("  Ví dụ: python step1_phonemize.py --ngan-dir D:/path/to/output_dataset")
        sys.exit(1)

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  PREPARE_NGAN — BƯỚC 1: PHONEMIZE")
    logger.info("=" * 60)
    logger.info(f"Config            : {config_path}")
    logger.info(f"Dataset dir       : {config.ngan_dataset_dir}")
    logger.info(f"Input filelists   : {config.input_filelists}")
    logger.info(f"Convert numbers   : {config.convert_numbers_to_words}")
    logger.info(f"Work dir          : {config.work_dir}")

    # --- Chạy ---
    try:
        main_process(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()