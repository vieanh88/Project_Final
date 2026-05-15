"""
=============================================================================
  BƯỚC 6: REBUILD MULTISPEAKER FILELIST
=============================================================================
Mục tiêu: Rebuild filelist ViVoice để hỗ trợ MULTISPEAKER training.

          Thay vì dùng cùng 1 speaker_id cho toàn bộ ViVoice, script này
          đọc lại cột `channel` từ dataset parquet, gán mỗi channel một
          speaker_id unique. Channels có < N samples được gộp vào
          speaker_id "other" (mặc định 999).

          KHÔNG cần chạy lại step2_extract_audio.py (tốn 6-8h).
          Script này tận dụng wav_paths.txt + phoneme_texts.txt đã có
          (thứ tự 1:1) và chỉ ghép thêm speaker_id từ cột channel.

Đầu vào : - Dataset ViVoice trong HF cache (đã tải ở Bước 1)
           - workdir/wav_paths.txt       (từ Bước 2)
           - workdir/phoneme_texts.txt   (từ Bước 3)

Đầu ra  : - output/vivoice_train_list.txt  (wav|phoneme|speaker_id)
           - output/vivoice_val_list.txt
           - output/speaker_id_map.json    (mapping channel ↔ speaker_id)

Format 1 dòng: wav_path|phoneme_text|speaker_id
  - Bác Ngạn     : speaker_id = 0 (đã được gán ở prepare_ngan)
  - ViVoice main : speaker_id = 1, 2, 3, ... (theo channel, count giảm dần)
  - ViVoice other: speaker_id = 999 (channels có < min_samples)

Chạy lệnh:
    python step6_rebuild_multispeaker.py
    python step6_rebuild_multispeaker.py --config config.yaml
    python step6_rebuild_multispeaker.py --min-samples 20 --force
=============================================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict
from collections import Counter

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
# CONSTANTS
# =============================================================================

# ID cố định cho các speaker đặc biệt
NGAN_SPEAKER_ID = 0               # Bác Ngạn (đã được prepare_ngan gán)
OTHER_SPEAKER_ID = 999            # Gộp các channels < min_samples
VIVOICE_START_ID = 1              # ViVoice channels bắt đầu từ ID này


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class RebuildConfig:
    """Cấu hình cho bước rebuild filelist multispeaker."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"
    cache_subdir: str = "hf_cache"

    # Dataset info (để load lại từ cache)
    dataset_name: str = "capleaf/viVoice"
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None

    # File input (từ step2 & step3)
    wav_paths_file: str = "wav_paths.txt"
    phoneme_text_file: str = "phoneme_texts.txt"

    # Tên cột channel trong dataset
    channel_column: str = "channel"

    # Threshold: channels < min_samples → gộp vào OTHER_SPEAKER_ID
    # (HOẶC drop hẳn nếu drop_below_threshold=True, xem dưới)
    min_samples_per_speaker: int = 20

    # Nếu True: DROP HẲN tất cả records của speakers < min_samples
    #   (KHÔNG gộp vào OTHER_SPEAKER_ID nữa)
    # Nếu False (default, giữ behavior cũ): gộp vào OTHER_SPEAKER_ID=999
    #
    # Use case "drop": train data sạch hơn, giảm dung lượng wav cần upload
    #   (vì wav của speakers bị drop sẽ KHÔNG xuất hiện trong filelist
    #    → có thể delete khỏi disk hoặc skip upload lên Vast.ai).
    drop_below_threshold: bool = False

    # Nếu > 0: CAP mỗi speaker ở tối đa N records (random sample với seed cố định)
    #   (chỉ áp dụng cho TRAIN records, KHÔNG áp dụng val)
    # Nếu = 0 (default): không cap, giữ nguyên phân phối tự nhiên
    #
    # Use case: dataset bị skewed (2-3 speakers chiếm 20-30% data) → cap để
    #   tránh model bias về những speakers đó + tiết kiệm dung lượng đáng kể.
    #
    # Ví dụ: cap_per_speaker=20000 → @VoizFM (99k records) chỉ giữ 20k
    #   → tiết kiệm ~80k records ≈ ~16 GB wav files.
    cap_per_speaker: int = 0

    # Train/Val split
    train_ratio: float = 0.95
    random_seed: int = 42

    # File output
    train_list: str = "vivoice_train_list.txt"
    val_list: str = "vivoice_val_list.txt"
    speaker_map_file: str = "speaker_id_map.json"

    # Delimiter
    delimiter: str = "|"

    # Validation
    min_phoneme_length: int = 3
    max_phoneme_length: int = 5000
    verify_wav_exists: bool = True

    # HF token
    hf_token: Optional[str] = None

    # Force overwrite
    force: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RebuildConfig":
        """Load config từ file YAML chung của prepare_vivoice."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step1 = full_config.get("step1_download", {})
        step3 = full_config.get("step3_phonemize", {})
        step5 = full_config.get("step5_filelist", {})
        step6 = full_config.get("step6_rebuild", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            cache_subdir=step1.get("cache_subdir", cls.cache_subdir),
            dataset_name=step1.get("dataset_name", cls.dataset_name),
            dataset_config=step1.get("dataset_config", cls.dataset_config),
            dataset_split=step1.get("dataset_split", cls.dataset_split),
            phoneme_text_file=step3.get("phoneme_text_file", cls.phoneme_text_file),
            channel_column=step6.get("channel_column", cls.channel_column),
            min_samples_per_speaker=step6.get("min_samples_per_speaker", cls.min_samples_per_speaker),
            drop_below_threshold=step6.get("drop_below_threshold", cls.drop_below_threshold),
            cap_per_speaker=step6.get("cap_per_speaker", cls.cap_per_speaker),
            train_ratio=step6.get("train_ratio", step5.get("train_ratio", cls.train_ratio)),
            random_seed=step6.get("random_seed", step5.get("random_seed", cls.random_seed)),
            train_list=step6.get("train_list", step5.get("train_list", cls.train_list)),
            val_list=step6.get("val_list", step5.get("val_list", cls.val_list)),
            speaker_map_file=step6.get("speaker_map_file", cls.speaker_map_file),
            delimiter=step6.get("delimiter", step5.get("delimiter", cls.delimiter)),
            min_phoneme_length=step6.get("min_phoneme_length", cls.min_phoneme_length),
            max_phoneme_length=step6.get("max_phoneme_length", cls.max_phoneme_length),
            verify_wav_exists=step6.get("verify_wav_exists", cls.verify_wav_exists),
        )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step6_rebuild_multispeaker.log"

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
    return logging.getLogger("step6_rebuild")


# =============================================================================
# CORE: EXTRACT CHANNELS FROM DATASET
# =============================================================================

def extract_channels_from_dataset(config: RebuildConfig, logger: logging.Logger) -> List[str]:
    """
    Duyệt lại dataset ViVoice theo ĐÚNG thứ tự step2_extract_audio.py đã duyệt,
    chỉ trích xuất cột `channel` (không load audio → CỰC NHANH).

    Returns:
        List[str]: channel[i] tương ứng với wav thứ i (vivoice_{i:07d}.wav)
    """
    from datasets import load_dataset
    from tqdm import tqdm

    cache_dir = Path(config.work_dir) / config.cache_subdir

    # Load dataset từ cache (giống step2)
    logger.info(f"Loading dataset từ cache: {cache_dir}")
    logger.info("(Chỉ đọc metadata cột `channel`, KHÔNG load audio → nhanh)")

    load_kwargs = {
        "path": config.dataset_name,
        "cache_dir": str(cache_dir)
    }
    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config
    if config.dataset_split:
        load_kwargs["split"] = config.dataset_split
    if config.hf_token:
        load_kwargs["token"] = config.hf_token

    dataset = load_dataset(**load_kwargs)

    # Xác định splits cần duyệt (GIỐNG step2)
    if hasattr(dataset, "keys"):
        splits = list(dataset.keys())
        logger.info(f"  Splits     : {splits}")
    else:
        splits = ["train"]
        dataset = {"train": dataset}

    # Kiểm tra cột channel có tồn tại không
    first_split = dataset[splits[0]]
    if config.channel_column not in first_split.column_names:
        logger.error(f"Cột '{config.channel_column}' không tồn tại trong dataset!")
        logger.error(f"Các cột có sẵn: {first_split.column_names}")
        raise KeyError(f"Column '{config.channel_column}' not found")

    # Duyệt theo đúng thứ tự step2
    # NOTE: Dùng .select_columns([channel]) để chỉ load cột channel (RẤT NHANH,
    # không phải decode audio bytes như step2)
    all_channels = []

    for split_name in splits:
        split_data = dataset[split_name].select_columns([config.channel_column])
        split_size = len(split_data)
        logger.info(f"  Split '{split_name}': {split_size:,} samples")

        # Duyệt tuần tự (PHẢI giữ đúng thứ tự index để khớp wav_paths.txt)
        for i in tqdm(range(split_size), desc=f"[{split_name}]", ncols=100):
            channel = split_data[i].get(config.channel_column, "unknown")
            # Normalize: strip whitespace, replace empty/None
            if not channel or not isinstance(channel, str):
                channel = "unknown"
            channel = channel.strip()
            if not channel:
                channel = "unknown"
            all_channels.append(channel)

    logger.info(f"Đã trích xuất {len(all_channels):,} channels")
    return all_channels


# =============================================================================
# CORE: BUILD SPEAKER_ID MAPPING
# =============================================================================

def build_speaker_id_map(
    channels: List[str],
    min_samples: int,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Tạo mapping channel → speaker_id theo Option B:
      - Channels có >= min_samples → speaker_id unique từ VIVOICE_START_ID
      - Channels có < min_samples  → speaker_id = OTHER_SPEAKER_ID

    Sắp xếp channels theo count giảm dần → channel lớn nhất có ID nhỏ nhất
    (thuận tiện cho debug).
    """
    channel_counter = Counter(channels)
    logger.info(f"Tổng số channels unique: {len(channel_counter):,}")

    # Phân loại channels
    main_channels = []
    other_channels = []

    for channel, count in channel_counter.items():
        if count >= min_samples:
            main_channels.append((channel, count))
        else:
            other_channels.append((channel, count))

    # Sort main channels theo count giảm dần → speaker_id tuần tự
    main_channels.sort(key=lambda x: (-x[1], x[0]))

    # Build mapping
    speaker_map: Dict[str, int] = {}
    for idx, (channel, count) in enumerate(main_channels):
        speaker_map[channel] = VIVOICE_START_ID + idx

    for channel, count in other_channels:
        speaker_map[channel] = OTHER_SPEAKER_ID

    # Log statistics
    logger.info(f"")
    logger.info(f"  Channels >= {min_samples} samples (main) : {len(main_channels):,}")
    logger.info(f"  Channels < {min_samples} samples (other) : {len(other_channels):,}")
    logger.info(f"  Other samples total                     : {sum(c for _, c in other_channels):,}")

    if main_channels:
        logger.info(f"")
        logger.info(f"  Top 15 main channels (theo count):")
        for channel, count in main_channels[:15]:
            sid = speaker_map[channel]
            logger.info(f"    speaker_id={sid:4d} | count={count:>8,} | {channel}")

        if len(main_channels) > 15:
            logger.info(f"    ... và {len(main_channels) - 15:,} channels khác")

    return speaker_map


# =============================================================================
# CORE: LOAD AUXILIARY FILES
# =============================================================================

def load_aux_files(config: RebuildConfig, logger: logging.Logger) -> tuple:
    """Đọc wav_paths.txt + phoneme_texts.txt (giữ index 1:1)."""
    work_dir = Path(config.work_dir)

    wav_paths_file = work_dir / config.wav_paths_file
    phoneme_file = work_dir / config.phoneme_text_file

    if not wav_paths_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {wav_paths_file}\n"
            f"Hãy chạy step2_extract_audio.py trước!"
        )

    if not phoneme_file.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {phoneme_file}\n"
            f"Hãy chạy step3_phonemize.py trước!"
        )

    with open(wav_paths_file, "r", encoding="utf-8") as f:
        wav_paths = [line.strip() for line in f.readlines()]

    with open(phoneme_file, "r", encoding="utf-8") as f:
        phonemes = [line.strip() for line in f.readlines()]

    logger.info(f"Wav paths  : {len(wav_paths):,} dòng")
    logger.info(f"Phonemes   : {len(phonemes):,} dòng")

    return wav_paths, phonemes


# =============================================================================
# CORE: ASSEMBLE & FILTER RECORDS
# =============================================================================

def assemble_records(
    wav_paths: List[str],
    phonemes: List[str],
    channels: List[str],
    speaker_map: Dict[str, int],
    config: RebuildConfig,
    logger: logging.Logger,
) -> List[str]:
    """
    Ghép wav_path + phoneme + speaker_id (lookup từ channel) thành records.
    Validate & filter.
    """
    # Kiểm tra số dòng khớp
    lengths = [len(wav_paths), len(phonemes), len(channels)]
    if len(set(lengths)) != 1:
        raise ValueError(
            f"Số dòng KHÔNG KHỚP: "
            f"wav_paths={len(wav_paths)}, "
            f"phonemes={len(phonemes)}, "
            f"channels={len(channels)}\n"
            f"Kiểm tra lại step2, step3, và dataset cache."
        )

    delim = config.delimiter
    valid_records = []

    stats = {
        "total": len(wav_paths),
        "valid": 0,
        "failed_phoneme": 0,
        "empty_phoneme": 0,
        "too_short": 0,
        "too_long": 0,
        "wav_missing": 0,
        "speaker_other": 0,           # Đếm số record THUỘC NHÓM OTHER (cả 2 mode)
        "dropped_below_threshold": 0, # Đếm số record bị DROP HẲN (mode drop=True)
    }

    # Bucket để log phân bố speaker_id
    speaker_record_counts: Dict[int, int] = {}

    for idx in range(len(wav_paths)):
        wav_path = wav_paths[idx]
        phoneme = phonemes[idx]
        channel = channels[idx]

        # Bỏ dòng phonemize thất bại
        if phoneme == "[FAILED]":
            stats["failed_phoneme"] += 1
            continue

        if not phoneme or phoneme.isspace():
            stats["empty_phoneme"] += 1
            continue

        if len(phoneme) < config.min_phoneme_length:
            stats["too_short"] += 1
            continue

        if len(phoneme) > config.max_phoneme_length:
            stats["too_long"] += 1
            continue

        if config.verify_wav_exists and not Path(wav_path).exists():
            stats["wav_missing"] += 1
            if stats["wav_missing"] <= 5:
                logger.warning(f"  Wav không tồn tại: {wav_path}")
            continue

        # Lookup speaker_id
        speaker_id = speaker_map.get(channel, OTHER_SPEAKER_ID)

        if speaker_id == OTHER_SPEAKER_ID:
            stats["speaker_other"] += 1

            # *** NEW: DROP HẲN nếu drop_below_threshold=True ***
            if config.drop_below_threshold:
                stats["dropped_below_threshold"] += 1
                continue  # KHÔNG ghi record này vào filelist

        speaker_record_counts[speaker_id] = speaker_record_counts.get(speaker_id, 0) + 1

        record = f"{wav_path}{delim}{phoneme}{delim}{speaker_id}"
        valid_records.append(record)
        stats["valid"] += 1

    # Log stats
    logger.info(f"")
    logger.info(f"Ghép records:")
    logger.info(f"  Tổng input          : {stats['total']:,}")
    logger.info(f"  Hợp lệ (kept)       : {stats['valid']:,}")
    logger.info(f"  Phoneme failed      : {stats['failed_phoneme']:,}")
    logger.info(f"  Phoneme rỗng        : {stats['empty_phoneme']:,}")
    logger.info(f"  Phoneme quá ngắn    : {stats['too_short']:,}")
    logger.info(f"  Phoneme quá dài     : {stats['too_long']:,}")
    logger.info(f"  Wav không tồn tại   : {stats['wav_missing']:,}")
    logger.info(f"  Thuộc nhóm OTHER    : {stats['speaker_other']:,}")
    if config.drop_below_threshold:
        logger.info(f"  → DROPPED (other)   : {stats['dropped_below_threshold']:,}  "
                    f"(do drop_below_threshold=True)")
    else:
        logger.info(f"  → Gộp vào id={OTHER_SPEAKER_ID}      : {stats['speaker_other']:,}  "
                    f"(drop_below_threshold=False)")

    # Log phân bố speaker_id (top 10 + OTHER)
    logger.info(f"")
    logger.info(f"  Phân bố records theo speaker_id:")
    sorted_speakers = sorted(
        speaker_record_counts.items(),
        key=lambda x: (-x[1], x[0]),
    )
    for sid, count in sorted_speakers[:10]:
        pct = count / stats["valid"] * 100 if stats["valid"] > 0 else 0
        label = "(OTHER)" if sid == OTHER_SPEAKER_ID else ""
        logger.info(f"    speaker_id={sid:4d}: {count:>8,} records ({pct:5.1f}%) {label}")

    if len(sorted_speakers) > 10:
        rest = sum(c for _, c in sorted_speakers[10:])
        logger.info(f"    ... và {len(sorted_speakers) - 10} speakers khác "
                    f"(tổng {rest:,} records)")

    return valid_records, speaker_record_counts


# =============================================================================
# CORE: CAP RECORDS PER SPEAKER (balanced subsampling)
# =============================================================================

def cap_records_per_speaker(
    records: List[str],
    cap: int,
    delimiter: str,
    seed: int,
    logger: logging.Logger,
) -> tuple:
    """
    Cap mỗi speaker ở tối đa `cap` records bằng random sampling.

    Mục đích: tránh model bị bias về các speakers chiếm % data lớn.
    Ví dụ: @VoizFM 99k records + @FonosVietnam 99k records (22.5% data).
    Cap = 20000 → mỗi speaker ≤ 20k records → phân phối đều hơn.

    Args:
        records: list các dòng "wav_path|phoneme|speaker_id"
        cap: ngưỡng tối đa records/speaker (> 0 mới active)
        delimiter: '|'
        seed: random seed (tách biệt với shuffle seed của train/val split)
        logger: logger

    Returns:
        (capped_records, new_speaker_counts) — list records đã cap, và dict
        speaker_id -> count mới.
    """
    if cap <= 0:
        # Không cap, return nguyên xi
        logger.info("Cap per speaker = 0 → bỏ qua bước cap (giữ nguyên phân phối).")
        # Tính lại count để return cho consistent
        counts = {}
        for r in records:
            parts = r.rsplit(delimiter, 1)
            if len(parts) == 2:
                try:
                    sid = int(parts[1])
                    counts[sid] = counts.get(sid, 0) + 1
                except ValueError:
                    pass
        return records, counts

    # Group records theo speaker_id
    grouped: Dict[int, List[str]] = {}
    for r in records:
        # speaker_id luôn là phần cuối, sau delimiter cuối
        parts = r.rsplit(delimiter, 1)
        if len(parts) != 2:
            logger.warning(f"Bỏ qua record format sai: {r[:80]}...")
            continue
        try:
            sid = int(parts[1])
        except ValueError:
            logger.warning(f"Bỏ qua record có speaker_id không phải int: {r[:80]}...")
            continue
        grouped.setdefault(sid, []).append(r)

    # Random sample mỗi nhóm
    rng = random.Random(seed)
    capped_records = []
    new_counts: Dict[int, int] = {}
    capped_speakers: List[tuple] = []  # (speaker_id, before, after)

    for sid in sorted(grouped.keys()):
        speaker_records = grouped[sid]
        before = len(speaker_records)

        if before <= cap:
            # Không cần cap
            capped_records.extend(speaker_records)
            new_counts[sid] = before
        else:
            # Random sample đúng `cap` records
            # Dùng rng.sample để đảm bảo reproducible
            sampled = rng.sample(speaker_records, cap)
            capped_records.extend(sampled)
            new_counts[sid] = cap
            capped_speakers.append((sid, before, cap))

    # Log chi tiết
    total_before = sum(len(v) for v in grouped.values())
    total_after = len(capped_records)
    saved = total_before - total_after

    logger.info(f"")
    logger.info(f"Cap per speaker = {cap:,}:")
    logger.info(f"  Records trước cap    : {total_before:,}")
    logger.info(f"  Records sau cap      : {total_after:,}")
    logger.info(f"  Tiết kiệm            : {saved:,} records "
                f"({saved / total_before * 100:.1f}%)")
    logger.info(f"  Speakers bị cap      : {len(capped_speakers)}")

    if capped_speakers:
        logger.info(f"")
        logger.info(f"  Chi tiết speakers bị cap:")
        for sid, before, after in capped_speakers:
            logger.info(f"    speaker_id={sid:4d}: {before:>8,} → {after:>8,}  "
                        f"(bỏ {before - after:,})")

    return capped_records, new_counts


# =============================================================================
# CORE: SHUFFLE & SPLIT
# =============================================================================

def shuffle_and_split(
    records: List[str],
    config: RebuildConfig,
    logger: logging.Logger,
) -> tuple:
    """Shuffle với seed cố định → split theo tỷ lệ train/val."""
    random.seed(config.random_seed)
    records_copy = records.copy()
    random.shuffle(records_copy)
    logger.info(f"Đã shuffle với seed={config.random_seed}")

    split_idx = int(len(records_copy) * config.train_ratio)
    train_records = records_copy[:split_idx]
    val_records = records_copy[split_idx:]

    logger.info(f"Split {config.train_ratio:.0%}/{1 - config.train_ratio:.0%}: "
                f"train={len(train_records):,}, val={len(val_records):,}")

    return train_records, val_records


# =============================================================================
# CORE: SAVE OUTPUTS
# =============================================================================

def save_outputs(
    train_records: List[str],
    val_records: List[str],
    speaker_map: Dict[str, int],
    speaker_record_counts: Dict[int, int],
    config: RebuildConfig,
    logger: logging.Logger,
):
    """Lưu 3 file: train_list, val_list, speaker_id_map.json."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list
    map_path = output_dir / config.speaker_map_file

    # Filelists
    with open(train_path, "w", encoding="utf-8") as f:
        for record in train_records:
            f.write(record + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for record in val_records:
            f.write(record + "\n")

    # Speaker ID map JSON (mapping + stats)
    # Reverse mapping: speaker_id → list of channels
    id_to_channels: Dict[int, List[str]] = {}
    for channel, sid in speaker_map.items():
        id_to_channels.setdefault(sid, []).append(channel)

    # Sort channels within each speaker_id
    for sid in id_to_channels:
        id_to_channels[sid].sort()

    map_data = {
        "_metadata": {
            "description": "Mapping channel ↔ speaker_id cho ViVoice multispeaker training",
            "min_samples_per_speaker": config.min_samples_per_speaker,
            "drop_below_threshold": config.drop_below_threshold,
            "cap_per_speaker": config.cap_per_speaker,
            "reserved_ids": {
                "ngan": NGAN_SPEAKER_ID,
                "other": OTHER_SPEAKER_ID,
                "vivoice_start": VIVOICE_START_ID,
            },
            "total_channels": len(speaker_map),
            "unique_speaker_ids": len(id_to_channels),
            "train_records": len(train_records),
            "val_records": len(val_records),
        },
        "channel_to_speaker_id": dict(sorted(speaker_map.items(), key=lambda x: (x[1], x[0]))),
        "speaker_id_to_channels": {
            str(sid): id_to_channels[sid]
            for sid in sorted(id_to_channels.keys())
        },
        "speaker_id_record_counts": {
            str(sid): speaker_record_counts.get(sid, 0)
            for sid in sorted(speaker_record_counts.keys())
        },
    }

    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(map_data, f, ensure_ascii=False, indent=2)

    logger.info(f"")
    logger.info(f"  Train file     : {train_path}")
    logger.info(f"  Val file       : {val_path}")
    logger.info(f"  Speaker map    : {map_path}")


# =============================================================================
# MAIN LOGIC
# =============================================================================

def rebuild_multispeaker(config: RebuildConfig, logger: logging.Logger):
    """Quy trình chính: load dataset → extract channels → build map → assemble → save."""
    output_dir = Path(config.output_dir)
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list

    # Check existing outputs
    if not config.force and train_path.exists() and val_path.exists():
        logger.info(f"Output đã tồn tại:")
        logger.info(f"  {train_path}")
        logger.info(f"  {val_path}")
        logger.info(f"Dùng --force để ghi đè, hoặc xóa file và chạy lại.")
        return

    start_time = time.time()

    # 1. Load dataset & extract channels
    logger.info("=" * 60)
    logger.info("  Bước 6.1: Trích xuất channels từ dataset")
    logger.info("=" * 60)
    channels = extract_channels_from_dataset(config, logger)

    # 2. Build speaker_id map
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.2: Xây dựng speaker_id mapping")
    logger.info("=" * 60)
    speaker_map = build_speaker_id_map(
        channels,
        config.min_samples_per_speaker,
        logger,
    )

    # 3. Load wav_paths + phonemes
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.3: Đọc wav_paths + phonemes")
    logger.info("=" * 60)
    wav_paths, phonemes = load_aux_files(config, logger)

    # 4. Assemble records
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.4: Ghép records (wav|phoneme|speaker_id)")
    logger.info("=" * 60)
    valid_records, speaker_record_counts = assemble_records(
        wav_paths, phonemes, channels,
        speaker_map, config, logger,
    )

    if not valid_records:
        logger.error("Không có record hợp lệ! Kiểm tra lại dữ liệu.")
        return

    # 4.5. Cap per speaker (NEW — balanced subsampling)
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.4.5: Cap records per speaker (balanced subsampling)")
    logger.info("=" * 60)
    # Dùng seed khác với train/val split để 2 phép random độc lập
    cap_seed = config.random_seed + 1
    valid_records, speaker_record_counts = cap_records_per_speaker(
        valid_records,
        config.cap_per_speaker,
        config.delimiter,
        cap_seed,
        logger,
    )

    # 5. Shuffle & split
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.5: Shuffle & split train/val")
    logger.info("=" * 60)
    train_records, val_records = shuffle_and_split(valid_records, config, logger)

    # 6. Save outputs
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6.6: Lưu outputs")
    logger.info("=" * 60)
    save_outputs(
        train_records, val_records,
        speaker_map, speaker_record_counts,
        config, logger,
    )

    elapsed = time.time() - start_time

    # Final summary
    logger.info("")
    logger.info("=" * 60)
    logger.info("  REBUILD MULTISPEAKER FILELIST HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"  Thời gian        : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Tổng records     : {len(valid_records):,}")
    logger.info(f"    Train          : {len(train_records):,}")
    logger.info(f"    Val            : {len(val_records):,}")
    logger.info(f"  Unique speakers  : {len(set(speaker_record_counts.keys()))}")
    logger.info(f"")
    logger.info(f"  Speaker ID ranges:")
    logger.info(f"    Bác Ngạn       : {NGAN_SPEAKER_ID} (trong prepare_ngan)")
    logger.info(f"    ViVoice main   : {VIVOICE_START_ID} → "
                f"{max((s for s in speaker_record_counts if s != OTHER_SPEAKER_ID), default=0)}")
    logger.info(f"    ViVoice other  : {OTHER_SPEAKER_ID}")
    logger.info(f"")
    logger.info("  Bước tiếp theo:")
    logger.info("    1. Sửa 3 file config (M2):")
    logger.info("       - configs/config_stage1.yaml: multispeaker: true")
    logger.info("       - configs/config_stage2.yaml: multispeaker: true")
    logger.info("       - configs/config_stage3.yaml: decoder.type → istftnet")
    logger.info("    2. Chạy train_wrapper.py --stage 1")
    logger.info("=" * 60)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bước 6: Rebuild ViVoice filelist cho multispeaker training"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=None,
        help="Override threshold (channels < N samples → speaker_id=OTHER, hoặc bị DROP)",
    )
    parser.add_argument(
        "--drop-below-threshold",
        action="store_true",
        help=(
            "DROP HẲN tất cả records của speakers < min_samples "
            "(thay vì gộp vào speaker_id=999). Khuyến nghị BẬT khi muốn "
            "filelist sạch + tiết kiệm dung lượng upload data."
        ),
    )
    parser.add_argument(
        "--cap-per-speaker",
        type=int,
        default=None,
        help=(
            "Cap mỗi speaker ở tối đa N records bằng random sampling. "
            "Mục đích: tránh model bias về 2-3 speakers chiếm % data lớn. "
            "Ví dụ --cap-per-speaker 20000 → giảm @VoizFM 99k → 20k. "
            "Default 0 = không cap."
        ),
    )
    parser.add_argument(
        "--no-verify-wav",
        action="store_true",
        help="Bỏ qua check .wav tồn tại (nhanh hơn)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè output cũ nếu đã tồn tại",
    )
    args = parser.parse_args()

    # Load .env
    env_candidates = [Path(".env"), Path("../.env"), Path("../../.env"), Path("../../../.env")]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(str(env_path))
            break

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = RebuildConfig.from_yaml(str(config_path))

    # HF token từ env
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if hf_token:
        config.hf_token = hf_token

    # Override từ CLI
    if args.min_samples is not None:
        config.min_samples_per_speaker = args.min_samples
    if args.drop_below_threshold:
        config.drop_below_threshold = True
    if args.cap_per_speaker is not None:
        config.cap_per_speaker = args.cap_per_speaker
    if args.no_verify_wav:
        config.verify_wav_exists = False
    if args.force:
        config.force = True

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  BƯỚC 6: REBUILD MULTISPEAKER FILELIST")
    logger.info("=" * 60)
    logger.info(f"Config               : {config_path.resolve()}")
    logger.info(f"Dataset              : {config.dataset_name}")
    logger.info(f"Cache dir            : {Path(config.work_dir) / config.cache_subdir}")
    logger.info(f"Channel column       : {config.channel_column}")
    logger.info(f"Min samples/speaker  : {config.min_samples_per_speaker}")
    logger.info(f"Drop below threshold : {config.drop_below_threshold}  "
                f"({'DROP HẲN' if config.drop_below_threshold else f'gộp vào id={OTHER_SPEAKER_ID}'})")
    logger.info(f"Cap per speaker      : {config.cap_per_speaker:,}  "
                f"({'ACTIVE' if config.cap_per_speaker > 0 else 'không cap'})")
    logger.info(f"Train ratio          : {config.train_ratio:.0%}")
    logger.info(f"Random seed          : {config.random_seed}")
    logger.info(f"Verify wav exists    : {config.verify_wav_exists}")
    logger.info(f"Force overwrite      : {config.force}")
    logger.info(f"Output dir           : {config.output_dir}")

    # Run
    try:
        rebuild_multispeaker(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()