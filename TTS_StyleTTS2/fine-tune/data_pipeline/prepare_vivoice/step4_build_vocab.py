"""
=============================================================================
  BƯỚC 4: BUILD VOCAB — Xây dựng từ điển âm vị (Phoneme Vocabulary)
=============================================================================
Mục tiêu: Quét toàn bộ file phoneme_texts.txt (tạo bởi Bước 3), đếm tất cả
          các ký tự IPA duy nhất (bao gồm khoảng trắng, dấu câu), tạo file
          phoneme_vocab.json chứa mapping ký_tự → ID và tổng n_token.

          Giá trị n_token sẽ được inject vào config YAML của StyleTTS2
          bởi train_wrapper.py để tránh lỗi IndexError.

Đầu vào : workdir/phoneme_texts.txt   (từ Bước 3)
Đầu ra  : output/phoneme_vocab.json

Chạy lệnh:
    python step4_build_vocab.py
    python step4_build_vocab.py --config config.yaml
    python step4_build_vocab.py --extra-phoneme-files path1.txt path2.txt
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from collections import Counter

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
# CONFIGURATION
# =============================================================================

@dataclass
class VocabConfig:
    """Cấu hình cho bước xây dựng từ điển phoneme."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # File phoneme chính (từ step3)
    phoneme_text_file: str = "phoneme_texts.txt"

    # File output
    vocab_file: str = "phoneme_vocab.json"

    # Có thêm special tokens không
    include_special_tokens: bool = True

    # Danh sách file phoneme bổ sung (từ prepare_ngan, prepare_ood)
    # Dùng để xây dựng vocab đầy đủ nhất ngay từ đầu
    extra_phoneme_files: List[str] = field(default_factory=list)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "VocabConfig":
        """Load config từ file YAML chung."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step3 = full_config.get("step3_phonemize", {})
        step4 = full_config.get("step4_vocab", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            phoneme_text_file=step3.get("phoneme_text_file", cls.phoneme_text_file),
            vocab_file=step4.get("vocab_file", cls.vocab_file),
            include_special_tokens=step4.get("include_special_tokens", cls.include_special_tokens),
        )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step4_build_vocab.log"

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
    return logging.getLogger("step4_vocab")


# =============================================================================
# CORE LOGIC
# =============================================================================

def scan_characters(file_path: Path, logger: logging.Logger) -> Counter:
    """
    Quét file phoneme, đếm tần suất từng ký tự IPA.
    Bỏ qua các dòng [FAILED].
    """
    char_counter = Counter()
    valid_lines = 0
    skipped_lines = 0

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "[FAILED]":
                skipped_lines += 1
                continue

            valid_lines += 1
            for char in line:
                char_counter[char] += 1

    logger.info(f"  {file_path.name}: {valid_lines:,} dòng hợp lệ, "
                f"{skipped_lines:,} dòng bỏ qua, "
                f"{len(char_counter):,} ký tự unique")

    return char_counter


def build_vocab(config: VocabConfig, logger: logging.Logger):
    """
    Quy trình chính: Quét phoneme files → xây dựng vocab → lưu JSON.
    """
    work_dir = Path(config.work_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Thu thập danh sách file cần quét ---
    phoneme_files = []

    # File chính từ step3
    main_file = work_dir / config.phoneme_text_file
    if main_file.exists():
        phoneme_files.append(main_file)
    else:
        logger.error(f"Không tìm thấy file phoneme chính: {main_file}")
        logger.error("Hãy chạy step3_phonemize.py trước!")
        return

    # File bổ sung (nếu có)
    for extra_path_str in config.extra_phoneme_files:
        extra_path = Path(extra_path_str)
        if extra_path.exists():
            phoneme_files.append(extra_path)
            logger.info(f"  Thêm file bổ sung: {extra_path}")
        else:
            logger.warning(f"  File bổ sung không tồn tại (bỏ qua): {extra_path}")

    logger.info(f"Tổng file cần quét: {len(phoneme_files)}")

    # --- Quét ký tự ---
    start_time = time.time()
    total_counter = Counter()

    for file_path in phoneme_files:
        file_counter = scan_characters(file_path, logger)
        total_counter.update(file_counter)

    # --- Xây dựng mapping ---
    # Sắp xếp theo tần suất giảm dần (ký tự phổ biến nhất có ID nhỏ nhất)
    sorted_chars = sorted(total_counter.keys(), key=lambda c: (-total_counter[c], c))

    # Special tokens (đặt ở đầu danh sách)
    special_tokens = {}
    next_id = 0

    if config.include_special_tokens:
        special_token_list = [
            ("<pad>", "Padding token"),
            ("<unk>", "Unknown token"),
            ("<bos>", "Begin of sequence"),
            ("<eos>", "End of sequence"),
        ]
        for token, desc in special_token_list:
            special_tokens[token] = next_id
            next_id += 1

    # Character mapping
    char_to_id = {}
    char_to_id.update(special_tokens)

    for char in sorted_chars:
        if char not in char_to_id:
            char_to_id[char] = next_id
            next_id += 1

    # Reverse mapping
    id_to_char = {v: k for k, v in char_to_id.items()}

    # n_token = tổng số entries trong vocab
    n_token = len(char_to_id)

    elapsed = time.time() - start_time

    # --- Tạo JSON output ---
    vocab_data = {
        "_metadata": {
            "description": "Phoneme vocabulary for StyleTTS2 Vietnamese",
            "n_token": n_token,
            "num_special_tokens": len(special_tokens),
            "num_phoneme_chars": n_token - len(special_tokens),
            "source_files": [str(f) for f in phoneme_files],
            "include_special_tokens": config.include_special_tokens,
        },
        "n_token": n_token,
        "char_to_id": char_to_id,
        "id_to_char": {str(k): v for k, v in id_to_char.items()},
        "char_frequencies": {
            char: total_counter[char]
            for char in sorted_chars
        },
    }

    # --- Lưu file ---
    vocab_path = output_dir / config.vocab_file
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ XÂY DỰNG VOCABULARY")
    logger.info("=" * 60)
    logger.info(f"  Thời gian        : {elapsed:.2f}s")
    logger.info(f"  n_token (TỔNG)   : {n_token}")
    logger.info(f"    Special tokens  : {len(special_tokens)}")
    logger.info(f"    Phoneme chars   : {n_token - len(special_tokens)}")
    logger.info(f"  Output file      : {vocab_path}")

    # In special tokens
    if special_tokens:
        logger.info("")
        logger.info("  Special tokens:")
        for token, tid in special_tokens.items():
            logger.info(f"    {tid:4d} → {token}")

    # In top 30 ký tự phổ biến nhất
    logger.info("")
    logger.info("  Top 30 ký tự phổ biến nhất:")
    for i, char in enumerate(sorted_chars[:30]):
        char_display = repr(char) if char in (' ', '\t', '\n') else char
        logger.info(
            f"    {char_to_id[char]:4d} → {char_display:6s}  "
            f"(xuất hiện {total_counter[char]:>10,} lần)"
        )

    # In bottom 10 ký tự hiếm nhất (có thể là noise)
    if len(sorted_chars) > 30:
        logger.info("")
        logger.info("  10 ký tự hiếm nhất (kiểm tra noise):")
        for char in sorted_chars[-10:]:
            char_display = repr(char) if char in (' ', '\t', '\n') else char
            logger.info(
                f"    {char_to_id[char]:4d} → {char_display:6s}  "
                f"(xuất hiện {total_counter[char]:>10,} lần)"
            )

    logger.info("")
    logger.info(f"  GIÁ TRỊ CẦN DÙNG: n_token = {n_token}")
    logger.info(f"  (Giá trị này sẽ được train_wrapper.py tự động inject vào config YAML)")
    logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bước 4: Xây dựng từ điển phoneme (Phoneme Vocabulary)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--extra-phoneme-files",
        nargs="*",
        default=[],
        help="Danh sách file phoneme bổ sung (từ prepare_ngan, prepare_ood) "
             "để xây dựng vocab đầy đủ nhất",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = VocabConfig.from_yaml(str(config_path))

    # Thêm extra files từ CLI
    if args.extra_phoneme_files:
        config.extra_phoneme_files.extend(args.extra_phoneme_files)

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  BƯỚC 4: XÂY DỰNG TỪ ĐIỂN PHONEME (VOCABULARY)")
    logger.info("=" * 60)
    logger.info(f"Config           : {config_path.resolve()}")
    logger.info(f"Phoneme file     : {Path(config.work_dir) / config.phoneme_text_file}")
    logger.info(f"Output vocab     : {Path(config.output_dir) / config.vocab_file}")
    logger.info(f"Special tokens   : {config.include_special_tokens}")
    if config.extra_phoneme_files:
        logger.info(f"Extra files      : {config.extra_phoneme_files}")

    # --- Chạy ---
    try:
        build_vocab(config, logger)
        logger.info("")
        logger.info("Bước tiếp theo: Chạy step5_make_filelist.py để tạo train/val filelist")
        logger.info("=" * 60)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.ex
if __name__ == "__main__":
    main()