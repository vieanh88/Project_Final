"""
=============================================================================
  BƯỚC 3: PHONEMIZE — Chuyển text thô tiếng Việt → chuỗi âm vị IPA
=============================================================================
Mục tiêu: Đọc file raw_texts.txt (tạo bởi Bước 2), chuyển từng dòng text
          sang chuỗi phoneme IPA bằng thư viện viphoneme, lưu ra file
          phoneme_texts.txt với index tương ứng 1:1.

Logic G2P: Tái sử dụng chính xác logic đã fix bug từ step08_phonemize.py
           (monkey-patch vinorm để tránh WinError 193 trên Windows 11).

Đầu vào : workdir/raw_texts.txt      (từ Bước 2)
Đầu ra  : workdir/phoneme_texts.txt  (1 dòng = 1 chuỗi IPA)
           workdir/phonemize_errors.txt (log các dòng lỗi)

Chạy lệnh:
    python step3_phonemize.py
    python step3_phonemize.py --config config.yaml
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

# KHẮC PHỤC LỖI ENCODING TRÊN WINDOWS
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

# Monkey-patch vinorm
vinorm.TTSnorm = _mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

# Import viphoneme SAU KHI đã patch vinorm
import viphoneme
viphoneme.TTSnorm = _mock_tts_norm

# Import hàm G2P cốt lõi
from viphoneme import vi2IPA_split

# CONFIGURATION
@dataclass
class PhonemizeConfig:
    """Cấu hình cho bước chuyển đổi text → phoneme."""

    # Đường dẫn
    work_dir: str = "./workdir"

    # File đầu vào (text thô, tạo bởi step2)
    raw_text_file: str = "raw_texts.txt"

    # File đầu ra (chuỗi phoneme IPA)
    phoneme_text_file: str = "phoneme_texts.txt"

    # Bỏ qua nếu file output đã tồn tại
    skip_existing: bool = True

    # Số dòng tối đa xử lý (0 = xử lý tất cả, dùng để test nhanh)
    max_lines: int = 0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PhonemizeConfig":
        """Load config từ file YAML chung."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step3 = full_config.get("step3_phonemize", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            raw_text_file=step3.get("raw_text_file", cls.raw_text_file),
            phoneme_text_file=step3.get("phoneme_text_file", cls.phoneme_text_file),
            skip_existing=step3.get("skip_existing", cls.skip_existing),
        )

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step3_phonemize.log"

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
    return logging.getLogger("step3_phonemize")

# CORE G2P FUNCTION
def clean_text(text: str) -> str:
    """
    Làm sạch text trước khi đưa vào G2P:
    - Loại bỏ ký tự đặc biệt không cần thiết
    - Giữ lại dấu câu cơ bản (, . ? !)
    - Chuẩn hóa khoảng trắng
    """
    # Loại bỏ các ký tự đặc biệt (giữ lại chữ cái, số, dấu câu cơ bản, khoảng trắng)
    # Giữ lại các ký tự tiếng Việt (Unicode range) và dấu câu
    text = re.sub(r'["""\'\(\)\[\]\{\}<>«»]', '', text)

    # Chuẩn hóa dấu câu liên tiếp
    text = re.sub(r'[.]{2,}', '.', text)
    text = re.sub(r'[!]{2,}', '!', text)
    text = re.sub(r'[?]{2,}', '?', text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def text_to_phoneme(text: str) -> str:
    """
    Chuyển đổi text tiếng Việt → chuỗi IPA phoneme.
    Logic giống hệt step08_phonemize.py đã fix bug.

    Returns:
        Chuỗi phoneme nếu thành công, "[ERROR] ..." nếu thất bại.
    """
    try:
        phonemes = vi2IPA_split(text, " ")
        # Chuẩn hóa khoảng trắng thừa
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes
    except Exception as e:
        return f"[ERROR] {str(e)}"

# CORE LOGIC
def phonemize_texts(config: PhonemizeConfig, logger: logging.Logger):
    """
    Quy trình chính: Đọc raw_texts.txt → chuyển phoneme → lưu phoneme_texts.txt.

    Đảm bảo index 1:1 giữa file input và output:
    - Dòng thành công → ghi chuỗi phoneme
    - Dòng thất bại → ghi "[FAILED]" (để step5 lọc bỏ, giữ nguyên thứ tự)
    """
    from tqdm import tqdm

    work_dir = Path(config.work_dir)

    # --- Đường dẫn file ---
    input_path = work_dir / config.raw_text_file
    output_path = work_dir / config.phoneme_text_file
    error_path = work_dir / "phonemize_errors.txt"

    # --- Kiểm tra file input ---
    if not input_path.exists():
        logger.error(f"Không tìm thấy file input: {input_path}")
        logger.error("Hãy chạy step2_extract_audio.py trước!")
        return

    # --- Skip nếu đã tồn tại ---
    if config.skip_existing and output_path.exists():
        line_count = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info(f"File phoneme đã tồn tại: {output_path} ({line_count:,} dòng)")
        logger.info("Bỏ qua (skip_existing=true). Xóa file nếu muốn chạy lại.")
        return

    # --- Đọc file input ---
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    total_lines = len(raw_lines)
    logger.info(f"Đã tải {total_lines:,} dòng text từ {input_path.name}")

    # Giới hạn nếu đang test
    if config.max_lines > 0:
        raw_lines = raw_lines[: config.max_lines]
        logger.info(f"Giới hạn xử lý: {config.max_lines} dòng (chế độ test)")

    # --- Xử lý phonemize ---
    phoneme_results = []
    error_records = []

    stats = {
        "success": 0,
        "failed": 0,
        "empty_input": 0,
        "empty_phoneme": 0,
    }

    start_time = time.time()

    for idx, line in enumerate(tqdm(raw_lines, desc="Phonemize", ncols=100)):
        text = line.strip()

        # Dòng trống
        if not text:
            phoneme_results.append("[FAILED]")
            stats["empty_input"] += 1
            continue

        # Làm sạch text
        cleaned = clean_text(text)
        if not cleaned:
            phoneme_results.append("[FAILED]")
            stats["empty_input"] += 1
            continue

        # Chuyển đổi G2P
        phoneme_str = text_to_phoneme(cleaned)

        if phoneme_str.startswith("[ERROR]"):
            phoneme_results.append("[FAILED]")
            stats["failed"] += 1
            error_records.append(f"Line {idx}: {text} → {phoneme_str}")
            if stats["failed"] <= 30:
                logger.warning(f"  Lỗi G2P dòng {idx}: {text[:60]}... → {phoneme_str}")
            continue

        # Kiểm tra kết quả rỗng
        if not phoneme_str or phoneme_str.isspace():
            phoneme_results.append("[FAILED]")
            stats["empty_phoneme"] += 1
            error_records.append(f"Line {idx}: {text} → (phoneme rỗng)")
            continue

        phoneme_results.append(phoneme_str)
        stats["success"] += 1

    elapsed = time.time() - start_time

    # --- Lưu file phoneme ---
    logger.info(f"Đang lưu {len(phoneme_results):,} dòng → {output_path}")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in phoneme_results:
            f.write(line + "\n")

    # --- Lưu file error log ---
    if error_records:
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"# Tổng lỗi: {len(error_records)}\n")
            f.write(f"# Thời gian: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            for record in error_records:
                f.write(record + "\n")
        logger.info(f"Chi tiết lỗi: {error_path}")

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ PHONEMIZE")
    logger.info("=" * 60)
    logger.info(f"  Thời gian       : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Tổng dòng       : {len(raw_lines):,}")
    logger.info(f"  Thành công      : {stats['success']:,}")
    logger.info(f"  Lỗi G2P        : {stats['failed']:,}")
    logger.info(f"  Text rỗng       : {stats['empty_input']:,}")
    logger.info(f"  Phoneme rỗng    : {stats['empty_phoneme']:,}")

    success_rate = (stats["success"] / len(raw_lines) * 100) if raw_lines else 0
    logger.info(f"  Tỷ lệ thành công: {success_rate:.1f}%")

    # --- Kiểm tra mẫu ---
    logger.info("")
    logger.info("  Mẫu kiểm tra (5 dòng đầu):")
    sample_count = 0
    for idx, (raw, phon) in enumerate(zip(raw_lines, phoneme_results)):
        if phon == "[FAILED]":
            continue
        raw_preview = raw.strip()[:50]
        phon_preview = phon[:60]
        logger.info(f"    [{idx}] {raw_preview}")
        logger.info(f"         → {phon_preview}")
        sample_count += 1
        if sample_count >= 5:
            break

    logger.info("")
    logger.info(f"  Output file : {output_path}")

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 3: Chuyển đổi text thô tiếng Việt → chuỗi phoneme IPA"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Giới hạn số dòng xử lý (0 = tất cả, dùng để test nhanh)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = PhonemizeConfig.from_yaml(str(config_path))

    # Override max_lines từ CLI nếu có
    if args.max_lines > 0:
        config.max_lines = args.max_lines

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("  BƯỚC 3: CHUYỂN ĐỔI TEXT → PHONEME IPA (G2P)")
    logger.info(f"Config         : {config_path.resolve()}")
    logger.info(f"Input file     : {Path(config.work_dir) / config.raw_text_file}")
    logger.info(f"Output file    : {Path(config.work_dir) / config.phoneme_text_file}")
    logger.info(f"Skip existing  : {config.skip_existing}")
    logger.info(f"G2P Engine     : viphoneme (vi2IPA_split)")

    # --- Chạy ---
    try:
        phonemize_texts(config, logger)
        logger.info("")
        logger.info("Bước tiếp theo: Chạy step4_build_vocab.py để xây dựng từ điển phoneme")
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()