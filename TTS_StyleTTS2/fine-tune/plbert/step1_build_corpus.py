"""
=============================================================================
  PL-BERT — BƯỚC 1: BUILD CORPUS
=============================================================================
Mục tiêu: Gộp toàn bộ text âm vị (CHỈ text phoneme, không gộp audio)
          từ 3 nguồn: ViVoice, Bác Ngạn, và OOD Text thành một file
          all_corpus_phoneme.txt để train PL-BERT tiếng Việt.

Nguồn dữ liệu:
  1. ViVoice filelist  (wav_path|phoneme|speaker_id) → trích cột phoneme
  2. Ngạn filelist     (wav_path|phoneme|speaker_id) → trích cột phoneme
  3. OOD phoneme       (chỉ có phoneme, không có wav)

Đầu ra:
  - all_corpus_phoneme.txt  (1 dòng = 1 chuỗi phoneme)
  - corpus_stats.json       (thống kê chi tiết)

Chạy lệnh:
    python step1_build_corpus.py --config config.yaml
    python step1_build_corpus.py \
        --vivoice-train "path/to/vivoice_train_list.txt" \
        --vivoice-val   "path/to/vivoice_val_list.txt" \
        --ngan-train    "path/to/ngan_train_list.txt" \
        --ngan-val      "path/to/ngan_val_list.txt" \
        --ood           "path/to/OOD_texts_phoneme.txt"
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
from typing import List, Optional
from collections import Counter

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
class CorpusConfig:
    """Cấu hình cho bước gộp corpus PL-BERT."""

    # Đường dẫn work & output
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # File output
    corpus_file: str = "all_corpus_phoneme.txt"
    stats_file: str = "corpus_stats.json"

    # --- Danh sách file nguồn ---
    # Filelist format (wav_path|phoneme|speaker_id) → trích cột phoneme (index 1)
    filelist_sources: List[str] = field(default_factory=list)

    # OOD format (chỉ có phoneme, không có delimiter)
    ood_sources: List[str] = field(default_factory=list)

    # Loại bỏ dòng trùng lặp
    deduplicate: bool = True

    # Shuffle corpus (quan trọng cho MLM training)
    shuffle: bool = True
    random_seed: int = 42

    # Lọc
    min_length: int = 5      # Bỏ dòng phoneme quá ngắn (ký tự)
    max_length: int = 10000  # Bỏ dòng phoneme quá dài

    # Bỏ qua nếu output đã tồn tại
    skip_existing: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "CorpusConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        plbert = full_config.get("plbert", {})

        return cls(
            work_dir=paths.get("work_dir", plbert.get("work_dir", cls.work_dir)),
            output_dir=paths.get("output_dir", plbert.get("output_dir", cls.output_dir)),
            corpus_file=plbert.get("corpus_file", cls.corpus_file),
            stats_file=plbert.get("stats_file", cls.stats_file),
            filelist_sources=plbert.get("filelist_sources", cls.filelist_sources),
            ood_sources=plbert.get("ood_sources", cls.ood_sources),
            deduplicate=plbert.get("deduplicate", cls.deduplicate),
            shuffle=plbert.get("shuffle", cls.shuffle),
            random_seed=plbert.get("random_seed", cls.random_seed),
            min_length=plbert.get("min_length", cls.min_length),
            max_length=plbert.get("max_length", cls.max_length),
            skip_existing=plbert.get("skip_existing", cls.skip_existing),
        )

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "plbert_step1_build_corpus.log"

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
    return logging.getLogger("plbert_step1")

# CORE LOGIC
def extract_phonemes_from_filelist(file_path: Path, logger: logging.Logger) -> List[str]:
    """
    Đọc filelist (wav_path|phoneme|speaker_id), trích cột phoneme.
    Hỗ trợ cả format 2 cột (wav_path|phoneme) và 3 cột.
    """
    phonemes = []
    skipped = 0

    if not file_path.exists():
        logger.warning(f"  Không tìm thấy: {file_path}")
        return phonemes

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("|")

            if len(parts) >= 2:
                # Cột phoneme là index 1
                phoneme = parts[1].strip()
                if phoneme and phoneme != "[FAILED]":
                    phonemes.append(phoneme)
                else:
                    skipped += 1
            else:
                skipped += 1

    logger.info(f"  {file_path.name}: {len(phonemes):,} phonemes, {skipped} bỏ qua")
    return phonemes

def extract_phonemes_from_ood(file_path: Path, logger: logging.Logger) -> List[str]:
    """
    Đọc file OOD (chỉ có phoneme, 1 dòng = 1 chuỗi phoneme).
    """
    phonemes = []
    skipped = 0

    if not file_path.exists():
        logger.warning(f"  Không tìm thấy: {file_path}")
        return phonemes

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line == "[FAILED]":
                skipped += 1
                continue
            phonemes.append(line)

    logger.info(f"  {file_path.name}: {len(phonemes):,} phonemes, {skipped} bỏ qua")
    return phonemes

def build_corpus(config: CorpusConfig, logger: logging.Logger):
    """
    Quy trình chính: Gộp phoneme từ tất cả nguồn → lọc → deduplicate → shuffle → lưu.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = output_dir / config.corpus_file

    # --- Skip nếu đã tồn tại ---
    if config.skip_existing and corpus_path.exists():
        line_count = sum(1 for _ in open(corpus_path, encoding="utf-8"))
        logger.info(f"Corpus đã tồn tại: {corpus_path} ({line_count:,} dòng) → bỏ qua")
        return

    start_time = time.time()

    # --- Thu thập phoneme từ các nguồn ---
    all_phonemes = []
    source_stats = {}

    # 1. Filelist sources (ViVoice, Ngạn)
    if config.filelist_sources:
        logger.info("")
        logger.info("Đọc filelist sources (wav_path|phoneme|speaker_id):")
        for src_path_str in config.filelist_sources:
            src_path = Path(src_path_str)
            phonemes = extract_phonemes_from_filelist(src_path, logger)
            source_stats[src_path.name] = len(phonemes)
            all_phonemes.extend(phonemes)

    # 2. OOD sources
    if config.ood_sources:
        logger.info("")
        logger.info("Đọc OOD sources (phoneme only):")
        for src_path_str in config.ood_sources:
            src_path = Path(src_path_str)
            phonemes = extract_phonemes_from_ood(src_path, logger)
            source_stats[src_path.name] = len(phonemes)
            all_phonemes.extend(phonemes)

    total_before_filter = len(all_phonemes)
    logger.info(f"\nTổng phoneme thu thập: {total_before_filter:,}")

    # --- Lọc theo độ dài ---
    filtered = [
        p for p in all_phonemes
        if config.min_length <= len(p) <= config.max_length
    ]
    filtered_count = total_before_filter - len(filtered)
    logger.info(f"Sau lọc độ dài [{config.min_length}, {config.max_length}]: "
                f"{len(filtered):,} (bỏ {filtered_count:,})")

    # --- Deduplicate ---
    before_dedup = len(filtered)
    if config.deduplicate:
        # Giữ thứ tự xuất hiện đầu tiên
        seen = set()
        unique = []
        for p in filtered:
            if p not in seen:
                seen.add(p)
                unique.append(p)
        filtered = unique
        dedup_removed = before_dedup - len(filtered)
        logger.info(f"Sau deduplicate: {len(filtered):,} (bỏ {dedup_removed:,} trùng lặp)")

    # --- Shuffle ---
    if config.shuffle:
        import random
        random.seed(config.random_seed)
        random.shuffle(filtered)
        logger.info(f"Đã shuffle với seed={config.random_seed}")

    # --- Lưu corpus ---
    with open(corpus_path, "w", encoding="utf-8") as f:
        for line in filtered:
            f.write(line + "\n")

    # --- Tính thống kê ---
    char_counter = Counter()
    lengths = []
    for line in filtered:
        char_counter.update(line)
        lengths.append(len(line))

    elapsed = time.time() - start_time

    # --- Lưu stats JSON ---
    stats_data = {
        "corpus_file": str(corpus_path),
        "total_lines": len(filtered),
        "total_characters": sum(lengths),
        "unique_characters": len(char_counter),
        "avg_length": sum(lengths) / len(lengths) if lengths else 0,
        "min_length": min(lengths) if lengths else 0,
        "max_length": max(lengths) if lengths else 0,
        "source_breakdown": source_stats,
        "deduplication": config.deduplicate,
        "shuffled": config.shuffle,
    }

    stats_path = output_dir / config.stats_file
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_data, f, ensure_ascii=False, indent=2)

    # --- Log kết quả ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ XÂY DỰNG CORPUS PL-BERT")
    logger.info("=" * 60)
    logger.info(f"  Thời gian          : {elapsed:.2f}s")
    logger.info(f"  Tổng dòng corpus   : {len(filtered):,}")
    logger.info(f"  Tổng ký tự         : {sum(lengths):,}")
    logger.info(f"  Ký tự unique       : {len(char_counter)}")
    logger.info(f"  Độ dài trung bình  : {stats_data['avg_length']:.1f} ký tự/dòng")
    logger.info("")
    logger.info("  Phân bổ theo nguồn:")
    for name, count in source_stats.items():
        pct = count / total_before_filter * 100 if total_before_filter > 0 else 0
        logger.info(f"    {name:40s}: {count:>10,}  ({pct:.1f}%)")
    logger.info("")
    logger.info(f"  Corpus file : {corpus_path}")
    logger.info(f"  Stats file  : {stats_path}")
    logger.info("")
    logger.info("  Bước tiếp theo: Chạy step2_train_plbert.py để train PL-BERT tiếng Việt")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="PL-BERT — Bước 1: Gộp corpus phoneme từ ViVoice + Ngạn + OOD"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    # Override trực tiếp từ CLI
    parser.add_argument("--vivoice-train", type=str, default=None)
    parser.add_argument("--vivoice-val", type=str, default=None)
    parser.add_argument("--ngan-train", type=str, default=None)
    parser.add_argument("--ngan-val", type=str, default=None)
    parser.add_argument("--ood", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if config_path.exists():
        config = CorpusConfig.from_yaml(str(config_path))
    else:
        config = CorpusConfig()

    # Override từ CLI
    cli_filelists = []
    cli_ood = []

    if args.vivoice_train:
        cli_filelists.append(args.vivoice_train)
    if args.vivoice_val:
        cli_filelists.append(args.vivoice_val)
    if args.ngan_train:
        cli_filelists.append(args.ngan_train)
    if args.ngan_val:
        cli_filelists.append(args.ngan_val)
    if args.ood:
        cli_ood.append(args.ood)

    if cli_filelists:
        config.filelist_sources = cli_filelists
    if cli_ood:
        config.ood_sources = cli_ood
    if args.output_dir:
        config.output_dir = args.output_dir

    # Kiểm tra có nguồn dữ liệu không
    if not config.filelist_sources and not config.ood_sources:
        print("[LỖI] Chưa chỉ định file nguồn dữ liệu!")
        print("  Cách 1: Dùng config.yaml với section 'plbert'")
        print("  Cách 2: Dùng CLI flags:")
        print("    python step1_build_corpus.py \\")
        print('      --vivoice-train "output/vivoice_train_list.txt" \\')
        print('      --ngan-train    "output/ngan_train_list.txt" \\')
        print('      --ood           "output/OOD_texts_phoneme.txt"')
        sys.exit(1)

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  PL-BERT — BƯỚC 1: XÂY DỰNG CORPUS")
    logger.info("=" * 60)
    logger.info(f"Config              : {config_path}")
    logger.info(f"Filelist sources    : {len(config.filelist_sources)} file(s)")
    for src in config.filelist_sources:
        logger.info(f"  → {src}")
    logger.info(f"OOD sources         : {len(config.ood_sources)} file(s)")
    for src in config.ood_sources:
        logger.info(f"  → {src}")
    logger.info(f"Deduplicate         : {config.deduplicate}")
    logger.info(f"Shuffle             : {config.shuffle}")
    logger.info(f"Output dir          : {config.output_dir}")

    # --- Chạy ---
    try:
        build_corpus(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()