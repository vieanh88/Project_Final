"""
=============================================================================
  BƯỚC 5: MAKE FILELIST — Tạo file train/val list chuẩn StyleTTS2
=============================================================================
Mục tiêu: Ghép 3 nguồn thông tin (wav_path, phoneme, speaker_id) thành
          filelist chuẩn StyleTTS2 và split train/val.

Đầu vào : - workdir/wav_paths.txt       (từ Bước 2)
           - workdir/phoneme_texts.txt   (từ Bước 3)
           - config.yaml (speaker_id, train_ratio, ...)

Đầu ra  : - output/vivoice_train_list.txt
           - output/vivoice_val_list.txt

Format 1 dòng: wav_path|phoneme_text|speaker_id
Ví dụ   : D:/project/wavs/vivoice_0000001.wav|s i n c a w f|1

Chạy lệnh:
    python step5_make_filelist.py
    python step5_make_filelist.py --config config.yaml
=============================================================================
"""

import os
import sys
import time
import random
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

# CONFIGURATION
@dataclass
class FilelistConfig:
    """Cấu hình cho bước tạo filelist train/val."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # File input (từ step2 & step3)
    wav_paths_file: str = "wav_paths.txt"
    phoneme_text_file: str = "phoneme_texts.txt"

    # Train/Val split
    train_ratio: float = 0.95
    random_seed: int = 42

    # File output
    train_list: str = "vivoice_train_list.txt"
    val_list: str = "vivoice_val_list.txt"

    # Speaker ID
    default_speaker_id: int = 1
    use_channel_as_speaker: bool = False

    # Delimiter
    delimiter: str = "|"

    # Validation
    min_phoneme_length: int = 3       # Bỏ phoneme quá ngắn (ký tự)
    max_phoneme_length: int = 5000    # Bỏ phoneme quá dài (ký tự)
    verify_wav_exists: bool = True    # Kiểm tra file .wav tồn tại trước khi ghi

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "FilelistConfig":
        """Load config từ file YAML chung."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step3 = full_config.get("step3_phonemize", {})
        step5 = full_config.get("step5_filelist", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            phoneme_text_file=step3.get("phoneme_text_file", cls.phoneme_text_file),
            train_ratio=step5.get("train_ratio", cls.train_ratio),
            random_seed=step5.get("random_seed", cls.random_seed),
            train_list=step5.get("train_list", cls.train_list),
            val_list=step5.get("val_list", cls.val_list),
            default_speaker_id=step5.get("default_speaker_id", cls.default_speaker_id),
            use_channel_as_speaker=step5.get("use_channel_as_speaker", cls.use_channel_as_speaker),
            delimiter=step5.get("delimiter", cls.delimiter),
        )

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step5_make_filelist.log"

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
    return logging.getLogger("step5_filelist")

# CORE LOGIC
def load_lines(file_path: Path) -> list:
    """Đọc file text, trả về list các dòng (giữ nguyên thứ tự)."""
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def make_filelist(config: FilelistConfig, logger: logging.Logger):
    """
    Quy trình chính:
    1. Đọc wav_paths.txt và phoneme_texts.txt
    2. Ghép 1:1 theo index, lọc bỏ dòng không hợp lệ
    3. Shuffle và split train/val
    4. Lưu filelist
    """
    work_dir = Path(config.work_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Đọc file input ---
    wav_paths_file = work_dir / config.wav_paths_file
    phoneme_file = work_dir / config.phoneme_text_file

    if not wav_paths_file.exists():
        logger.error(f"Không tìm thấy: {wav_paths_file}")
        logger.error("Hãy chạy step2_extract_audio.py trước!")
        return

    if not phoneme_file.exists():
        logger.error(f"Không tìm thấy: {phoneme_file}")
        logger.error("Hãy chạy step3_phonemize.py trước!")
        return

    wav_paths = load_lines(wav_paths_file)
    phonemes = load_lines(phoneme_file)

    logger.info(f"Wav paths  : {len(wav_paths):,} dòng")
    logger.info(f"Phonemes   : {len(phonemes):,} dòng")

    # --- Kiểm tra số dòng khớp ---
    if len(wav_paths) != len(phonemes):
        logger.error(
            f"Số dòng KHÔNG KHỚP: wav_paths={len(wav_paths)}, phonemes={len(phonemes)}"
        )
        logger.error("Kiểm tra lại Bước 2 và Bước 3. Hai file phải có cùng số dòng.")
        return

    # --- Ghép và lọc ---
    valid_records = []
    delim = config.delimiter
    speaker_id = config.default_speaker_id

    stats = {
        "total": len(wav_paths),
        "valid": 0,
        "failed_phoneme": 0,
        "empty_phoneme": 0,
        "too_short": 0,
        "too_long": 0,
        "wav_missing": 0,
    }

    start_time = time.time()

    for idx in range(len(wav_paths)):
        wav_path = wav_paths[idx]
        phoneme = phonemes[idx]

        # Bỏ dòng phonemize thất bại
        if phoneme == "[FAILED]":
            stats["failed_phoneme"] += 1
            continue

        # Bỏ phoneme rỗng
        if not phoneme or phoneme.isspace():
            stats["empty_phoneme"] += 1
            continue

        # Bỏ phoneme quá ngắn
        if len(phoneme) < config.min_phoneme_length:
            stats["too_short"] += 1
            continue

        # Bỏ phoneme quá dài
        if len(phoneme) > config.max_phoneme_length:
            stats["too_long"] += 1
            continue

        # Kiểm tra file .wav tồn tại
        if config.verify_wav_exists and not Path(wav_path).exists():
            stats["wav_missing"] += 1
            if stats["wav_missing"] <= 10:
                logger.warning(f"  Wav không tồn tại: {wav_path}")
            continue

        # Tạo dòng filelist: wav_path|phoneme|speaker_id
        record = f"{wav_path}{delim}{phoneme}{delim}{speaker_id}"
        valid_records.append(record)
        stats["valid"] += 1

    logger.info(f"Ghép xong: {stats['valid']:,} records hợp lệ / {stats['total']:,} tổng")

    if stats["valid"] == 0:
        logger.error("Không có record hợp lệ nào! Kiểm tra lại dữ liệu.")
        return

    # --- Shuffle ---
    random.seed(config.random_seed)
    random.shuffle(valid_records)
    logger.info(f"Đã shuffle với seed={config.random_seed}")

    # --- Split train/val ---
    split_idx = int(len(valid_records) * config.train_ratio)
    train_records = valid_records[:split_idx]
    val_records = valid_records[split_idx:]

    logger.info(f"Split {config.train_ratio:.0%}/{1 - config.train_ratio:.0%}: "
                f"train={len(train_records):,}, val={len(val_records):,}")

    # --- Lưu file ---
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list

    with open(train_path, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(record + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(record + "\n")

    elapsed = time.time() - start_time

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ TẠO FILELIST")
    logger.info("=" * 60)
    logger.info(f"  Thời gian         : {elapsed:.2f}s")
    logger.info(f"  Tổng records      : {stats['total']:,}")
    logger.info(f"  Hợp lệ           : {stats['valid']:,}")
    logger.info(f"  Phoneme thất bại  : {stats['failed_phoneme']:,}")
    logger.info(f"  Phoneme rỗng      : {stats['empty_phoneme']:,}")
    logger.info(f"  Phoneme quá ngắn  : {stats['too_short']:,}")
    logger.info(f"  Phoneme quá dài   : {stats['too_long']:,}")
    logger.info(f"  Wav không tồn tại : {stats['wav_missing']:,}")
    logger.info("")
    logger.info(f"  Train file : {train_path}  ({len(train_records):,} records)")
    logger.info(f"  Val file   : {val_path}  ({len(val_records):,} records)")
    logger.info(f"  Speaker ID : {speaker_id}")

    # --- In mẫu ---
    logger.info("")
    logger.info("  Mẫu kiểm tra (3 dòng train đầu):")
    for i, record in enumerate(train_records[:3]):
        parts = record.split(delim)
        wav_name = Path(parts[0]).name if len(parts) > 0 else "?"
        phon_preview = parts[1][:50] + "..." if len(parts) > 1 and len(parts[1]) > 50 else parts[1] if len(parts) > 1 else "?"
        sid = parts[2] if len(parts) > 2 else "?"
        logger.info(f"    [{i}] {wav_name} | {phon_preview} | {sid}")

    # --- Tính thống kê độ dài phoneme ---
    phoneme_lengths = []
    for record in valid_records:
        parts = record.split(delim)
        if len(parts) >= 2:
            phoneme_lengths.append(len(parts[1]))

    if phoneme_lengths:
        import statistics
        logger.info("")
        logger.info("  Thống kê độ dài phoneme (ký tự):")
        logger.info(f"    Min    : {min(phoneme_lengths)}")
        logger.info(f"    Max    : {max(phoneme_lengths)}")
        logger.info(f"    Mean   : {statistics.mean(phoneme_lengths):.1f}")
        logger.info(f"    Median : {statistics.median(phoneme_lengths):.1f}")
        logger.info(f"    Stdev  : {statistics.stdev(phoneme_lengths):.1f}" if len(phoneme_lengths) > 1 else "")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE PREPARE_VIVOICE HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info("  Các file output:")
    logger.info(f"    1. Wav files     : {Path(config.output_dir)}/vivoice_clean_wavs/")
    logger.info(f"    2. Phoneme vocab : {Path(config.output_dir)}/phoneme_vocab.json")
    logger.info(f"    3. Train list    : {train_path}")
    logger.info(f"    4. Val list      : {val_path}")
    logger.info("")
    logger.info("  Bước tiếp theo trong Roadmap:")
    logger.info("    → prepare_ngan/step1_phonemize.py  (Phonemize dataset Bác Ngạn)")
    logger.info("    → prepare_ood/step1_clean_phonemize.py  (Clean + phonemize OOD text)")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 5: Tạo file train/val list chuẩn StyleTTS2"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--no-verify-wav",
        action="store_true",
        help="Bỏ qua việc kiểm tra file .wav tồn tại (nhanh hơn nhưng kém an toàn)",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = FilelistConfig.from_yaml(str(config_path))

    if args.no_verify_wav:
        config.verify_wav_exists = False

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  BƯỚC 5: TẠO FILELIST TRAIN/VAL CHUẨN STYLETTS2")
    logger.info("=" * 60)
    logger.info(f"Config          : {config_path.resolve()}")
    logger.info(f"Wav paths file  : {Path(config.work_dir) / config.wav_paths_file}")
    logger.info(f"Phoneme file    : {Path(config.work_dir) / config.phoneme_text_file}")
    logger.info(f"Train ratio     : {config.train_ratio:.0%}")
    logger.info(f"Speaker ID      : {config.default_speaker_id}")
    logger.info(f"Verify wav      : {config.verify_wav_exists}")
    logger.info(f"Output dir      : {config.output_dir}")

    # --- Chạy ---
    try:
        make_filelist(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()