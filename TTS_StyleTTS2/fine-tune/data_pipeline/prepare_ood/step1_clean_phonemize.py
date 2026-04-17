"""
=============================================================================
  PREPARE_OOD — BƯỚC 1: CLEAN & PHONEMIZE
=============================================================================
Mục tiêu: Đọc file raw_ood_texts.txt (50,000 câu truyện ma thô), làm sạch
          (lọc số, ký tự đặc biệt), chuyển sang phoneme IPA, lưu ra file
          OOD_texts_phoneme.txt cho Joint Adversarial Training (JAT).

          File OOD này KHÔNG có audio đi kèm — chỉ có text phoneme.
          Nó được dùng ở Giai đoạn 2 & 3 để dạy Prosodic Encoder dự đoán
          nhịp điệu cho từ vựng domain truyện ma.

Đầu vào : raw_ood_texts.txt  (1 dòng = 1 câu hoặc 1 đoạn văn)
Đầu ra  : OOD_texts_phoneme.txt  (1 dòng = 1 chuỗi phoneme IPA)

Chạy lệnh:
    python step1_clean_phonemize.py --input "D:/path/to/raw_ood_texts.txt"
    python step1_clean_phonemize.py --config config.yaml
    python step1_clean_phonemize.py --input raw.txt --max-lines 100
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
from typing import Optional

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
# =============================================================================
import vinorm

def _mock_tts_norm(text, *args, **kwargs):
    return str(text).lower().strip()

vinorm.TTSnorm = _mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

import viphoneme
viphoneme.TTSnorm = _mock_tts_norm

from viphoneme import vi2IPA_split


# =============================================================================
# CONFIGURATION
# =============================================================================

# Bảng chuyển số → chữ tiếng Việt
DIGIT_TO_WORD = {
    "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
    "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
}


@dataclass
class OODConfig:
    """Cấu hình cho bước clean & phonemize OOD text."""

    # File input (50k câu truyện ma thô)
    input_file: str = ""

    # Thư mục work (log, file trung gian)
    work_dir: str = "./workdir"

    # Thư mục output
    output_dir: str = "./output"

    # Tên file output
    output_file: str = "OOD_texts_phoneme.txt"

    # --- Tùy chọn Clean ---
    # Chuyển số thành chữ (true) hoặc xóa hoàn toàn (false + remove_numbers=true)
    convert_numbers_to_words: bool = True
    remove_numbers: bool = False

    # Độ dài câu cho phép (theo từ, SAU khi clean)
    min_words: int = 3       # Bỏ câu quá ngắn
    max_words: int = 60      # Bỏ câu quá dài (tránh OOM khi train)

    # --- Tùy chọn Split ---
    # Tách đoạn văn dài thành nhiều câu ngắn (theo dấu . ? !)
    split_long_sentences: bool = True
    split_max_words: int = 25  # Nếu câu > N từ, thử tách tại dấu câu

    # Số dòng tối đa (0 = tất cả)
    max_lines: int = 0

    # Bỏ qua nếu output đã tồn tại
    skip_existing: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OODConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        ood = full_config.get("prepare_ood", {})

        return cls(
            input_file=ood.get("input_file", cls.input_file),
            work_dir=paths.get("work_dir", ood.get("work_dir", cls.work_dir)),
            output_dir=paths.get("output_dir", ood.get("output_dir", cls.output_dir)),
            output_file=ood.get("output_file", cls.output_file),
            convert_numbers_to_words=ood.get("convert_numbers_to_words", cls.convert_numbers_to_words),
            remove_numbers=ood.get("remove_numbers", cls.remove_numbers),
            min_words=ood.get("min_words", cls.min_words),
            max_words=ood.get("max_words", cls.max_words),
            split_long_sentences=ood.get("split_long_sentences", cls.split_long_sentences),
            split_max_words=ood.get("split_max_words", cls.split_max_words),
            max_lines=ood.get("max_lines", cls.max_lines),
            skip_existing=ood.get("skip_existing", cls.skip_existing),
        )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ood_step1_clean_phonemize.log"

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
    return logging.getLogger("ood_step1")


# =============================================================================
# TEXT CLEANING — Đặc biệt cho OOD truyện ma
# =============================================================================

def digits_to_words(text: str) -> str:
    """Chuyển từng chữ số thành chữ tiếng Việt."""
    result = []
    for char in text:
        if char.isdigit():
            result.append(DIGIT_TO_WORD.get(char, char))
            result.append(" ")
        else:
            result.append(char)
    return "".join(result)


def clean_ood_text(text: str, convert_numbers: bool = True, remove_numbers: bool = False) -> str:
    """
    Làm sạch text OOD truyện ma:
    1. Loại bỏ ngoặc kép, ngoặc đơn, các ký tự rườm rà
    2. Chỉ giữ dấu câu cơ bản: , . ? !
    3. Xử lý số
    4. Chuẩn hóa khoảng trắng
    """
    # Xóa ký tự đặc biệt rườm rà
    text = re.sub(r'["""\'\'\(\)\[\]\{\}<>«»—–_\-\+\=\*\#\@\&\^\~\`\\\/\|;:]', ' ', text)

    # Chuẩn hóa dấu câu liên tiếp
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    # Xử lý số
    if convert_numbers:
        text = digits_to_words(text)
    elif remove_numbers:
        text = re.sub(r'\d+', '', text)

    # Chỉ giữ ký tự chữ, dấu câu cơ bản, khoảng trắng
    text = re.sub(r'[^\w\s,.\?!]', ' ', text)
    text = text.replace('_', ' ')

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def split_into_sentences(text: str, max_words: int = 25) -> list:
    """
    Tách đoạn văn dài thành nhiều câu ngắn hơn.
    Tách tại dấu . ? ! nếu câu vượt quá max_words.
    """
    # Tách sơ bộ theo dấu câu
    # Giữ lại dấu câu ở cuối mỗi phần
    raw_parts = re.split(r'(?<=[.?!])\s+', text)

    sentences = []
    buffer = ""

    for part in raw_parts:
        part = part.strip()
        if not part:
            continue

        if not buffer:
            buffer = part
        else:
            # Kiểm tra nếu gộp vào có quá dài không
            combined = buffer + " " + part
            if len(combined.split()) <= max_words:
                buffer = combined
            else:
                # Buffer đã đủ dài, lưu lại
                sentences.append(buffer)
                buffer = part

    # Phần còn lại
    if buffer:
        sentences.append(buffer)

    return sentences


def text_to_phoneme(text: str) -> str:
    """Chuyển đổi text → phoneme IPA."""
    try:
        phonemes = vi2IPA_split(text, " ")
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes
    except Exception as e:
        return f"[ERROR] {str(e)}"


# =============================================================================
# CORE LOGIC
# =============================================================================

def process_ood_texts(config: OODConfig, logger: logging.Logger):
    """
    Quy trình chính: Đọc raw text → clean → split → phonemize → lưu file.
    """
    from tqdm import tqdm

    input_path = Path(config.input_file)
    work_dir = Path(config.work_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / config.output_file

    # --- Kiểm tra input ---
    if not input_path.exists():
        logger.error(f"Không tìm thấy file input: {input_path}")
        logger.error("Chỉ định đường dẫn bằng --input hoặc trong config.yaml (prepare_ood.input_file)")
        return

    # --- Skip nếu đã tồn tại ---
    if config.skip_existing and output_path.exists():
        line_count = sum(1 for _ in open(output_path, encoding="utf-8"))
        logger.info(f"Output đã tồn tại: {output_path} ({line_count:,} dòng) → bỏ qua")
        logger.info("Xóa file nếu muốn chạy lại.")
        return

    # --- Đọc file input ---
    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    total_raw = len(raw_lines)
    logger.info(f"Đã tải {total_raw:,} dòng từ {input_path.name}")

    if config.max_lines > 0:
        raw_lines = raw_lines[: config.max_lines]
        logger.info(f"Giới hạn: {config.max_lines} dòng (chế độ test)")

    # --- PHASE 1: Clean & Split ---
    logger.info("")
    logger.info("Phase 1: Clean & Split text...")

    cleaned_sentences = []
    stats = {
        "raw_lines": len(raw_lines),
        "after_clean": 0,
        "after_split": 0,
        "too_short": 0,
        "too_long": 0,
        "phoneme_ok": 0,
        "phoneme_error": 0,
        "phoneme_empty": 0,
    }

    for line in tqdm(raw_lines, desc="Clean", ncols=100):
        text = line.strip()
        if not text:
            continue

        # Clean
        cleaned = clean_ood_text(
            text,
            convert_numbers=config.convert_numbers_to_words,
            remove_numbers=config.remove_numbers,
        )

        if not cleaned:
            continue

        stats["after_clean"] += 1

        # Split nếu quá dài
        if config.split_long_sentences and len(cleaned.split()) > config.split_max_words:
            sub_sentences = split_into_sentences(cleaned, config.split_max_words)
        else:
            sub_sentences = [cleaned]

        for sent in sub_sentences:
            word_count = len(sent.split())

            if word_count < config.min_words:
                stats["too_short"] += 1
                continue

            if word_count > config.max_words:
                stats["too_long"] += 1
                continue

            cleaned_sentences.append(sent)

    stats["after_split"] = len(cleaned_sentences)
    logger.info(f"Sau clean & split: {stats['after_split']:,} câu")

    # --- Lưu file clean tạm (debug) ---
    clean_temp_path = work_dir / "ood_cleaned_texts.txt"
    with open(clean_temp_path, "w", encoding="utf-8") as f:
        for sent in cleaned_sentences:
            f.write(sent + "\n")
    logger.info(f"Saved clean text (debug): {clean_temp_path}")

    # --- PHASE 2: Phonemize ---
    logger.info("")
    logger.info("Phase 2: Phonemize...")

    phoneme_results = []
    error_records = []

    start_time = time.time()

    for sent in tqdm(cleaned_sentences, desc="Phonemize", ncols=100):
        phoneme_str = text_to_phoneme(sent)

        if phoneme_str.startswith("[ERROR]"):
            stats["phoneme_error"] += 1
            error_records.append(f"{sent} → {phoneme_str}")
            if stats["phoneme_error"] <= 20:
                logger.warning(f"  Lỗi G2P: {sent[:50]}...")
            continue

        if not phoneme_str or phoneme_str.isspace():
            stats["phoneme_empty"] += 1
            continue

        phoneme_results.append(phoneme_str)
        stats["phoneme_ok"] += 1

    elapsed = time.time() - start_time

    # --- Lưu file output ---
    with open(output_path, "w", encoding="utf-8") as f:
        for line in phoneme_results:
            f.write(line + "\n")

    # Lưu error log
    if error_records:
        error_path = work_dir / "ood_phonemize_errors.txt"
        with open(error_path, "w", encoding="utf-8") as f:
            f.write(f"# Tổng lỗi: {len(error_records)}\n\n")
            for record in error_records:
                f.write(record + "\n")
        logger.info(f"Chi tiết lỗi: {error_path}")

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ CLEAN & PHONEMIZE OOD TEXT")
    logger.info("=" * 60)
    logger.info(f"  Thời gian        : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Dòng raw         : {stats['raw_lines']:,}")
    logger.info(f"  Sau clean        : {stats['after_clean']:,}")
    logger.info(f"  Sau split        : {stats['after_split']:,}")
    logger.info(f"    Quá ngắn       : {stats['too_short']:,}")
    logger.info(f"    Quá dài        : {stats['too_long']:,}")
    logger.info(f"  Phoneme OK       : {stats['phoneme_ok']:,}")
    logger.info(f"  Phoneme lỗi      : {stats['phoneme_error']:,}")
    logger.info(f"  Phoneme rỗng     : {stats['phoneme_empty']:,}")

    success_rate = stats["phoneme_ok"] / stats["after_split"] * 100 if stats["after_split"] > 0 else 0
    logger.info(f"  Tỷ lệ thành công : {success_rate:.1f}%")
    logger.info("")
    logger.info(f"  Output file      : {output_path}")
    logger.info(f"  Tổng dòng output : {len(phoneme_results):,}")

    # Mẫu kiểm tra
    logger.info("")
    logger.info("  Mẫu kiểm tra (5 dòng đầu):")
    for i in range(min(5, len(phoneme_results))):
        raw_preview = cleaned_sentences[i][:40] if i < len(cleaned_sentences) else "?"
        phon_preview = phoneme_results[i][:50]
        logger.info(f"    [{i}] {raw_preview}")
        logger.info(f"         → {phon_preview}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE PREPARE_OOD HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info("  File OOD sẽ được trỏ vào config_stage2.yaml và config_stage3.yaml")
    logger.info("  tại trường: data_params.OOD_data")
    logger.info("")
    logger.info("  Bước tiếp theo trong Roadmap:")
    logger.info("    → plbert/step1_build_corpus.py  (Gộp corpus cho PL-BERT)")
    logger.info("    → Hoặc rebuild vocab nếu cần:")
    logger.info(f"      python step4_build_vocab.py --extra-phoneme-files {output_path}")
    logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Prepare OOD — Clean & Phonemize 50k câu truyện ma cho JAT"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default=None,
        help="Override đường dẫn file raw_ood_texts.txt",
    )
    parser.add_argument(
        "--max-lines",
        type=int,
        default=0,
        help="Giới hạn số dòng (0 = tất cả)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if config_path.exists():
        config = OODConfig.from_yaml(str(config_path))
    else:
        config = OODConfig()

    # Override từ CLI
    if args.input:
        config.input_file = args.input
    if args.max_lines > 0:
        config.max_lines = args.max_lines

    # Kiểm tra bắt buộc
    if not config.input_file:
        print("[LỖI] Chưa chỉ định file input OOD text!")
        print("  Dùng --input hoặc đặt 'prepare_ood.input_file' trong config.yaml")
        print("  Ví dụ: python step1_clean_phonemize.py --input D:/path/to/raw_ood_texts.txt")
        sys.exit(1)

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  PREPARE_OOD — CLEAN & PHONEMIZE TRUYỆN MA")
    logger.info("=" * 60)
    logger.info(f"Config            : {config_path}")
    logger.info(f"Input file        : {config.input_file}")
    logger.info(f"Output file       : {Path(config.output_dir) / config.output_file}")
    logger.info(f"Convert numbers   : {config.convert_numbers_to_words}")
    logger.info(f"Split sentences   : {config.split_long_sentences} (max {config.split_max_words} từ)")
    logger.info(f"Word range        : [{config.min_words}, {config.max_words}]")

    # --- Chạy ---
    try:
        process_ood_texts(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()