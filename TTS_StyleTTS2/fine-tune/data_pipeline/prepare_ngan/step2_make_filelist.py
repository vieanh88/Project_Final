"""
=============================================================================
  PREPARE_NGAN — BƯỚC 2: MAKE FILELIST
=============================================================================
Mục tiêu: Đọc file phoneme đã tạo ở Bước 1, append speaker_id = 0
          (Bác Ngạn), validate từng record, và split train/val.

Đầu vào : workdir/ngan_train_phoneme.txt  (wav_path|phoneme, từ Bước 1)
           workdir/ngan_val_phoneme.txt

Đầu ra  : output/ngan_train_list.txt      (wav_path|phoneme|0)
           output/ngan_val_list.txt        (wav_path|phoneme|0)

           Đồng thời kiểm tra chéo phoneme vocab (nếu có) để phát hiện
           ký tự IPA mới chưa có trong từ điển ViVoice.

Chạy lệnh:
    python step2_make_filelist.py
    python step2_make_filelist.py --config config.yaml
    python step2_make_filelist.py --ngan-dir "D:/path/to/output_dataset"
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
from typing import Optional, List

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
class NganFilelistConfig:
    """Cấu hình cho bước tạo filelist Bác Ngạn."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # Đường dẫn thư mục chứa wav Bác Ngạn (để verify file tồn tại)
    ngan_dataset_dir: str = ""

    # File phoneme input (từ step1, nằm trong work_dir)
    # [FIX] Dùng field(default_factory=...) thay vì None để dataclass hoạt động đúng
    phoneme_files: List[str] = field(default_factory=lambda: [
        "ngan_train_phoneme.txt",
        "ngan_val_phoneme.txt",
    ])

    # File output
    train_list: str = "ngan_train_list.txt"
    val_list: str = "ngan_val_list.txt"

    # Speaker ID cho Bác Ngạn (0 = Ngạn, 1 = ViVoice)
    speaker_id: int = 0

    # Đường dẫn phoneme_vocab.json (để kiểm tra chéo, tùy chọn)
    vocab_file: Optional[str] = None

    # Validation
    min_phoneme_length: int = 3
    max_phoneme_length: int = 5000
    verify_wav_exists: bool = True

    # Split (chỉ áp dụng nếu input chỉ có 1 file duy nhất)
    # Nếu input đã có sẵn train/val riêng → giữ nguyên, không re-split
    train_ratio: float = 0.95
    random_seed: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "NganFilelistConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        ngan = full_config.get("prepare_ngan", {})

        # [FIX] Load phoneme_files từ YAML nếu có khai báo, 
        #       ngược lại dùng giá trị mặc định của class
        default_phoneme_files = [
            "ngan_train_phoneme.txt",
            "ngan_val_phoneme.txt",
        ]
        phoneme_files = ngan.get("phoneme_files", default_phoneme_files)
    
        return cls(
            work_dir=paths.get("work_dir", ngan.get("work_dir", cls.work_dir)),         # ưu tiên paths.work_dir > prepare_ngan.work_dir > default
            output_dir=paths.get("output_dir", ngan.get("output_dir", cls.output_dir)), # ưu tiên paths.output_dir > prepare_ngan.output_dir > default
            ngan_dataset_dir=ngan.get("dataset_dir", cls.ngan_dataset_dir),             # chỉ có trong prepare_ngan
            phoneme_files=phoneme_files,  # [FIX] Thêm dòng này
            speaker_id=ngan.get("speaker_id", cls.speaker_id),                          # chỉ có trong prepare_ngan
            vocab_file=ngan.get("vocab_file", paths.get("vocab_file", cls.vocab_file)), # ưu tiên prepare_ngan.vocab_file > paths.vocab_file > default
            verify_wav_exists=ngan.get("verify_wav_exists", cls.verify_wav_exists),     # chỉ có trong prepare_ngan
            train_ratio=ngan.get("train_ratio", cls.train_ratio),                       # chỉ có trong prepare_ngan
            random_seed=ngan.get("random_seed", cls.random_seed),                       # chỉ có trong prepare_ngan
        )

# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "ngan_step2_make_filelist.log"

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
    return logging.getLogger("ngan_step2")


# =============================================================================
# VOCAB CROSS-CHECK
# =============================================================================

def load_vocab(vocab_path: Path, logger: logging.Logger) -> dict:
    """Load phoneme_vocab.json để kiểm tra chéo."""
    if not vocab_path.exists():
        logger.warning(f"Vocab file không tồn tại: {vocab_path} → bỏ qua kiểm tra chéo")
        return {}

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    char_to_id = vocab_data.get("char_to_id", {})
    logger.info(f"Loaded vocab: {len(char_to_id)} ký tự từ {vocab_path.name}")
    return char_to_id


def check_oov_characters(phoneme: str, char_to_id: dict) -> list:
    """Tìm các ký tự trong phoneme chưa có trong vocab (OOV)."""
    oov_chars = []
    for char in phoneme:
        if char not in char_to_id:
            oov_chars.append(char)
    return oov_chars


# =============================================================================
# CORE LOGIC
# =============================================================================

def process_phoneme_file(
    input_path: Path,
    config: NganFilelistConfig,
    char_to_id: dict,
    logger: logging.Logger,
) -> list:
    """
    Đọc file phoneme (wav_path|phoneme), validate, append speaker_id.

    Returns:
        List các record hợp lệ: wav_path|phoneme|speaker_id
    """
    if not input_path.exists():
        logger.warning(f"Bỏ qua (không tìm thấy): {input_path}")
        return []

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đọc {len(lines):,} dòng từ {input_path.name}")

    valid_records = []
    oov_counter = {}
    stats = {
        "total": 0,
        "valid": 0,
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "wav_missing": 0,
        "bad_format": 0,
    }

    for line in lines:
        line = line.strip()
        if not line:
            continue

        stats["total"] += 1
        parts = line.split("|")

        # Expect format: wav_path|phoneme
        if len(parts) < 2:
            stats["bad_format"] += 1
            continue

        wav_path = parts[0].strip()
        phoneme = parts[1].strip()

        # Phoneme rỗng
        if not phoneme or phoneme.isspace():
            stats["empty"] += 1
            continue

        # Phoneme quá ngắn
        if len(phoneme) < config.min_phoneme_length:
            stats["too_short"] += 1
            continue

        # Phoneme quá dài
        if len(phoneme) > config.max_phoneme_length:
            stats["too_long"] += 1
            continue

        # Kiểm tra wav tồn tại
        if config.verify_wav_exists and wav_path:
            wav_full = Path(wav_path)
            if not wav_full.is_absolute():
                # Thử ghép với ngan_dataset_dir
                wav_full = Path(config.ngan_dataset_dir) / wav_path
            if not wav_full.exists():
                stats["wav_missing"] += 1
                if stats["wav_missing"] <= 10:
                    logger.warning(f"  Wav không tồn tại: {wav_full}")
                continue

        # Kiểm tra OOV (nếu có vocab)
        if char_to_id:
            oov_chars = check_oov_characters(phoneme, char_to_id)
            for c in oov_chars:
                oov_counter[c] = oov_counter.get(c, 0) + 1

        # Append speaker_id
        record = f"{wav_path}|{phoneme}|{config.speaker_id}"
        valid_records.append(record)
        stats["valid"] += 1

    # Log stats cho file này
    logger.info(
        f"  Kết quả {input_path.name}: "
        f"valid={stats['valid']:,} | "
        f"empty={stats['empty']} | "
        f"short={stats['too_short']} | "
        f"long={stats['too_long']} | "
        f"wav_missing={stats['wav_missing']} | "
        f"bad_format={stats['bad_format']}"
    )

    # Log OOV characters
    if oov_counter:
        logger.warning(f"  Phát hiện {len(oov_counter)} ký tự OOV (chưa có trong vocab):")
        for char, count in sorted(oov_counter.items(), key=lambda x: -x[1])[:20]:
            char_display = repr(char)
            logger.warning(f"    {char_display}: xuất hiện {count:,} lần")
        logger.warning(
            "  → Cần rebuild vocab (step4_build_vocab.py --extra-phoneme-files ...) "
            "để bổ sung các ký tự này!"
        )

    return valid_records


def make_ngan_filelist(config: NganFilelistConfig, logger: logging.Logger):
    """
    Quy trình chính: Đọc phoneme files → validate → append speaker_id → lưu filelist.
    """
    work_dir = Path(config.work_dir)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Load vocab (tùy chọn) ---
    char_to_id = {}
    if config.vocab_file:
        vocab_path = Path(config.vocab_file)
        if not vocab_path.is_absolute():
            vocab_path = output_dir / config.vocab_file
        char_to_id = load_vocab(vocab_path, logger)

    # --- Xử lý từng file phoneme ---
    start_time = time.time()
    all_train_records = []
    all_val_records = []

    for filename in config.phoneme_files:
        input_path = work_dir / filename

        records = process_phoneme_file(input_path, config, char_to_id, logger)

        # Phân loại train/val dựa trên tên file
        if "val" in filename.lower():
            all_val_records.extend(records)
        else:
            all_train_records.extend(records)

    # --- Nếu chỉ có train (không có val riêng) → split ---
    if all_train_records and not all_val_records:
        import random
        random.seed(config.random_seed)
        random.shuffle(all_train_records)

        split_idx = int(len(all_train_records) * config.train_ratio)
        all_val_records = all_train_records[split_idx:]
        all_train_records = all_train_records[:split_idx]

        logger.info(
            f"Auto-split {config.train_ratio:.0%}/{1 - config.train_ratio:.0%}: "
            f"train={len(all_train_records):,}, val={len(all_val_records):,}"
        )

    # --- Lưu file ---
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list

    with open(train_path, "w", encoding="utf-8") as f:
        for record in all_train_records:
            f.write(record + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for record in all_val_records:
            f.write(record + "\n")

    elapsed = time.time() - start_time

    # --- Thống kê tổng ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ TẠO FILELIST BÁC NGẠN")
    logger.info("=" * 60)
    logger.info(f"  Thời gian      : {elapsed:.2f}s")
    logger.info(f"  Speaker ID     : {config.speaker_id} (Bác Ngạn)")
    logger.info(f"  Train records  : {len(all_train_records):,}")
    logger.info(f"  Val records    : {len(all_val_records):,}")
    logger.info(f"  Tổng           : {len(all_train_records) + len(all_val_records):,}")
    logger.info("")
    logger.info(f"  Train file     : {train_path}")
    logger.info(f"  Val file       : {val_path}")

    # In mẫu
    logger.info("")
    logger.info("  Mẫu train (3 dòng đầu):")
    for i, record in enumerate(all_train_records[:3]):
        parts = record.split("|")
        wav_name = Path(parts[0]).name if parts[0] else "?"
        phon = parts[1][:45] + "..." if len(parts[1]) > 45 else parts[1]
        sid = parts[2]
        logger.info(f"    [{i}] {wav_name} | {phon} | {sid}")

    # Tính thống kê audio
    if all_train_records:
        import statistics
        phon_lengths = [
            len(r.split("|")[1])
            for r in all_train_records
            if len(r.split("|")) >= 2
        ]
        if phon_lengths:
            logger.info("")
            logger.info("  Thống kê độ dài phoneme (ký tự):")
            logger.info(f"    Min    : {min(phon_lengths)}")
            logger.info(f"    Max    : {max(phon_lengths)}")
            logger.info(f"    Mean   : {statistics.mean(phon_lengths):.1f}")
            logger.info(f"    Median : {statistics.median(phon_lengths):.1f}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  PIPELINE PREPARE_NGAN HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info("  Bước tiếp theo trong Roadmap:")
    logger.info("    → prepare_ood/step1_clean_phonemize.py  (Clean + phonemize OOD text)")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Prepare Ngạn — Bước 2: Tạo filelist train/val với speaker_id"
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
        help="Override đường dẫn thư mục output_dataset Bác Ngạn",
    )
    parser.add_argument(
        "--no-verify-wav",
        action="store_true",
        help="Bỏ qua kiểm tra file .wav tồn tại",
    )
    args = parser.parse_args()

    # --- Load config ---
    config_path = Path(args.config)
    if config_path.exists():
        config = NganFilelistConfig.from_yaml(str(config_path))
    else:
        config = NganFilelistConfig()

    # Override từ CLI
    if args.ngan_dir:
        config.ngan_dataset_dir = args.ngan_dir
    if args.no_verify_wav:
        config.verify_wav_exists = False

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  PREPARE_NGAN — BƯỚC 2: TẠO FILELIST")
    logger.info("=" * 60)
    logger.info(f"Config         : {config_path}")
    logger.info(f"Dataset dir    : {config.ngan_dataset_dir}")
    logger.info(f"Speaker ID     : {config.speaker_id}")
    logger.info(f"Verify wav     : {config.verify_wav_exists}")
    logger.info(f"Vocab file     : {config.vocab_file or '(không dùng)'}")
    logger.info(f"Output dir     : {config.output_dir}")
    logger.info(f"Phoneme files  : {config.phoneme_files}")

    # --- Chạy ---
    try:
        make_ngan_filelist(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()