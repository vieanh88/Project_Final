"""
=============================================================================
  BƯỚC 6B: APPLY FILTERS TO EXISTING FILELISTS
=============================================================================
Mục tiêu: Áp dụng MIN_SAMPLES filter + CAP_PER_SPEAKER trực tiếp trên
          filelist đã tồn tại (output của step6), KHÔNG đụng đến HF
          dataset hay parquet cache. Chạy < 1 phút.

Use case:
  - step6_rebuild_multispeaker.py đã chạy với min_samples=20
  - Đã có output/vivoice_train_list.txt + vivoice_val_list.txt + speaker_id_map.json
  - Đã có ~887k wav files trong output/vivoice_clean_wavs/
  - Bây giờ muốn filter mạnh hơn (min_samples=1000) + cap (20000)
    để giảm số wav cần upload Vast.ai

Logic ( gộp train+val rồi filter+cap+split lại):
  1. Đọc 2 filelist hiện tại → gộp thành 1 pool
  2. Đếm records/speaker_id từ pool (KHÔNG dựa vào speaker_id_record_counts cũ
     trong speaker_id_map.json, để đảm bảo nhất quán với filelist thực tế)
  3. Apply min_samples filter: DROP records của speakers < min_samples
     (KHÔNG gộp vào id=999 nữa)
  4. Apply cap_per_speaker: random sample mỗi speaker xuống ≤ cap
     (chỉ áp dụng cho TỔNG pool, KHÔNG cap riêng train/val)
  5. Shuffle với random_seed → split 95/5
  6. Lưu:
     - vivoice_train_list.txt   (overwrite)
     - vivoice_val_list.txt     (overwrite)
     - speaker_id_map.json      (overwrite, GIỮ NGUYÊN channel_to_speaker_id
       vì mapping channel→id là invariant; chỉ update metadata + record counts)
     - wavs_to_delete.txt        (danh sách wav cần XÓA khỏi disk)
     - wavs_to_keep.txt          (danh sách wav cần GIỮ + upload Drive)
  7. Backup speaker_id_map.json → speaker_id_map.json.backup_before_filter
     (chỉ 1 lần duy nhất, không overwrite backup nếu đã có)

Đầu vào : - output/vivoice_train_list.txt   (từ step6 cũ)
           - output/vivoice_val_list.txt     (từ step6 cũ)
           - output/speaker_id_map.json      (từ step6 cũ, để giữ channel mapping)

Đầu ra  : - output/vivoice_train_list.txt   (overwrite)
           - output/vivoice_val_list.txt     (overwrite)
           - output/speaker_id_map.json      (overwrite metadata)
           - output/speaker_id_map.json.backup_before_filter  (chỉ tạo 1 lần)
           - output/wavs_to_delete.txt        (full paths, mỗi dòng 1 file)
           - output/wavs_to_keep.txt          (full paths, mỗi dòng 1 file)
           - workdir/logs/step6b_apply_filters.log

Chạy lệnh:
    python step6b_apply_filters.py
    python step6b_apply_filters.py --min-samples 1000 --cap-per-speaker 20000
    python step6b_apply_filters.py --dry-run
=============================================================================
"""

import os
import sys
import json
import time
import random
import logging
import argparse
import shutil
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Dict, Tuple
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
class ApplyFiltersConfig:
    """Cấu hình cho step6b apply filters."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # File input/output (đều trong output_dir)
    train_list: str = "vivoice_train_list.txt"
    val_list: str = "vivoice_val_list.txt"
    speaker_map_file: str = "speaker_id_map.json"

    # File output mới
    wavs_to_delete_file: str = "wavs_to_delete.txt"
    wavs_to_keep_file: str = "wavs_to_keep.txt"

    # Filter params
    min_samples_per_speaker: int = 1000
    cap_per_speaker: int = 20000

    # Train/Val split
    train_ratio: float = 0.95
    random_seed: int = 42

    # Delimiter cho filelist
    delimiter: str = "|"

    # Dry-run: log thống kê nhưng KHÔNG ghi file
    dry_run: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ApplyFiltersConfig":
        """Load config từ file YAML chung của prepare_vivoice."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step5 = full_config.get("step5_filelist", {})
        step6 = full_config.get("step6_rebuild", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            train_list=step6.get("train_list", step5.get("train_list", cls.train_list)),
            val_list=step6.get("val_list", step5.get("val_list", cls.val_list)),
            speaker_map_file=step6.get("speaker_map_file", cls.speaker_map_file),
            min_samples_per_speaker=step6.get(
                "min_samples_per_speaker", cls.min_samples_per_speaker
            ),
            cap_per_speaker=step6.get("cap_per_speaker", cls.cap_per_speaker),
            train_ratio=step6.get(
                "train_ratio", step5.get("train_ratio", cls.train_ratio)
            ),
            random_seed=step6.get(
                "random_seed", step5.get("random_seed", cls.random_seed)
            ),
            delimiter=step6.get("delimiter", step5.get("delimiter", cls.delimiter)),
        )

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step6b_apply_filters.log"

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
    return logging.getLogger("step6b_apply_filters")

# CORE: PARSE RECORDS
def parse_record(line: str, delimiter: str) -> Optional[Tuple[str, str, int]]:
    """
    Parse 1 dòng filelist → (wav_path, phoneme, speaker_id).
    Return None nếu dòng không hợp lệ.
    """
    line = line.strip()
    if not line:
        return None
    # Format: wav_path|phoneme|speaker_id
    # CHÚ Ý: phoneme có thể chứa nhiều token nhưng KHÔNG chứa "|" (đã clean ở step3)
    # Dùng rsplit để lấy speaker_id (luôn là phần cuối)
    parts = line.rsplit(delimiter, 1)
    if len(parts) != 2:
        return None
    try:
        speaker_id = int(parts[1])
    except ValueError:
        return None
    # Phần trước = wav_path + delimiter + phoneme
    left_parts = parts[0].split(delimiter, 1)
    if len(left_parts) != 2:
        return None
    wav_path, phoneme = left_parts[0], left_parts[1]
    return (wav_path, phoneme, speaker_id)


def load_filelist(file_path: Path, delimiter: str, logger: logging.Logger) -> List[Tuple[str, str, int]]:
    """Đọc filelist → list of (wav_path, phoneme, speaker_id)."""
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy: {file_path}")

    records = []
    skipped = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            parsed = parse_record(line, delimiter)
            if parsed is None:
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"  Bỏ qua dòng {line_idx} không hợp lệ: {line.strip()[:80]}")
                continue
            records.append(parsed)
    logger.info(f"  {file_path.name}: {len(records):,} records (bỏ qua {skipped})")
    return records

# CORE: FILTER + CAP
def filter_and_cap(
    records: List[Tuple[str, str, int]],
    min_samples: int,
    cap: int,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Tuple[str, str, int]], Dict[int, int], Dict[int, int]]:
    """
    Apply filter (drop speakers < min_samples) + cap (mỗi speaker ≤ cap).

    Args:
        records: list (wav_path, phoneme, speaker_id) — pool gộp train+val
        min_samples: ngưỡng speakers cần có (records < ngưỡng → DROP)
        cap: cap tối đa records/speaker (0 = không cap)
        seed: random seed cho sampling
        logger: logger

    Returns:
        (filtered_records, counts_before, counts_after)
    """
    # Đếm records/speaker_id từ pool gộp
    counts_before: Dict[int, int] = Counter(r[2] for r in records)
    logger.info(f"")
    logger.info(f"Phân bố records BAN ĐẦU (gộp train+val):")
    logger.info(f"  Tổng records      : {len(records):,}")
    logger.info(f"  Unique speakers   : {len(counts_before)}")

    sorted_speakers = sorted(counts_before.items(), key=lambda x: -x[1])
    logger.info(f"  Top 5:")
    for sid, c in sorted_speakers[:5]:
        logger.info(f"    speaker_id={sid:4d}: {c:>8,} records")
    logger.info(f"  Bottom 5:")
    for sid, c in sorted_speakers[-5:]:
        logger.info(f"    speaker_id={sid:4d}: {c:>8,} records")

    # =====================================================================
    # STEP 1: Filter speakers < min_samples
    # =====================================================================
    kept_speaker_ids = {sid for sid, c in counts_before.items() if c >= min_samples}
    dropped_speaker_ids = {sid for sid, c in counts_before.items() if c < min_samples}

    logger.info(f"")
    logger.info(f"STEP 1: Filter speakers có < {min_samples:,} records")
    logger.info(f"  Speakers GIỮ      : {len(kept_speaker_ids)}")
    logger.info(f"  Speakers DROP     : {len(dropped_speaker_ids)}")

    dropped_records_count = sum(counts_before[sid] for sid in dropped_speaker_ids)
    logger.info(f"  Records sẽ drop   : {dropped_records_count:,}")

    if dropped_speaker_ids:
        # Log 5 speakers bị drop có nhiều records nhất (để verify đúng)
        dropped_sorted = sorted(
            [(sid, counts_before[sid]) for sid in dropped_speaker_ids],
            key=lambda x: -x[1],
        )
        logger.info(f"  Top speakers bị drop (max 5):")
        for sid, c in dropped_sorted[:5]:
            logger.info(f"    speaker_id={sid:4d}: {c:>8,} records")

    # Filter
    records_after_filter = [r for r in records if r[2] in kept_speaker_ids]
    logger.info(f"  Sau filter        : {len(records_after_filter):,} records "
                f"(giảm {len(records) - len(records_after_filter):,})")

    # =====================================================================
    # STEP 2: Cap per speaker
    # =====================================================================
    if cap <= 0:
        logger.info(f"")
        logger.info(f"STEP 2: cap_per_speaker=0 → BỎ QUA cap")
        final_records = records_after_filter
    else:
        logger.info(f"")
        logger.info(f"STEP 2: Cap mỗi speaker ở {cap:,} records (random sample, seed={seed})")

        # Group records theo speaker_id
        grouped: Dict[int, List[Tuple[str, str, int]]] = {}
        for r in records_after_filter:
            grouped.setdefault(r[2], []).append(r)

        # Random sample mỗi nhóm
        rng = random.Random(seed)
        final_records = []
        capped_speakers: List[Tuple[int, int, int]] = []  # (sid, before, after)

        for sid in sorted(grouped.keys()):
            speaker_records = grouped[sid]
            before = len(speaker_records)
            if before <= cap:
                final_records.extend(speaker_records)
            else:
                sampled = rng.sample(speaker_records, cap)
                final_records.extend(sampled)
                capped_speakers.append((sid, before, cap))

        logger.info(f"  Speakers bị cap   : {len(capped_speakers)}")
        if capped_speakers:
            logger.info(f"  Chi tiết:")
            for sid, before, after in capped_speakers:
                saved = before - after
                logger.info(f"    speaker_id={sid:4d}: {before:>8,} → {after:>8,}  "
                            f"(bỏ {saved:,})")

        capped_total_saved = sum(b - a for _, b, a in capped_speakers)
        logger.info(f"  Sau cap           : {len(final_records):,} records "
                    f"(tiết kiệm thêm {capped_total_saved:,})")

    # =====================================================================
    # FINAL counts
    # =====================================================================
    counts_after: Dict[int, int] = Counter(r[2] for r in final_records)

    logger.info(f"")
    logger.info(f"KẾT QUẢ filter + cap:")
    logger.info(f"  Records ban đầu   : {len(records):,}")
    logger.info(f"  Records cuối cùng : {len(final_records):,}")
    logger.info(f"  Tổng tiết kiệm    : {len(records) - len(final_records):,} "
                f"({(len(records) - len(final_records)) / len(records) * 100:.1f}%)")
    logger.info(f"  Speakers cuối cùng: {len(counts_after)}")

    return final_records, dict(counts_before), dict(counts_after)

# CORE: SHUFFLE + SPLIT
def shuffle_and_split(
    records: List[Tuple[str, str, int]],
    train_ratio: float,
    seed: int,
    logger: logging.Logger,
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[str, str, int]]]:
    """Shuffle với seed cố định → split theo train_ratio."""
    rng = random.Random(seed)
    records_copy = records.copy()
    rng.shuffle(records_copy)

    split_idx = int(len(records_copy) * train_ratio)
    train_records = records_copy[:split_idx]
    val_records = records_copy[split_idx:]

    logger.info(f"")
    logger.info(f"Shuffle + Split {train_ratio:.0%}/{1-train_ratio:.0%} (seed={seed}):")
    logger.info(f"  Train: {len(train_records):,}")
    logger.info(f"  Val  : {len(val_records):,}")

    # Verify val có đủ diverse speakers
    val_speaker_set = {r[2] for r in val_records}
    train_speaker_set = {r[2] for r in train_records}
    logger.info(f"  Train speakers: {len(train_speaker_set)}")
    logger.info(f"  Val speakers  : {len(val_speaker_set)}")
    missing_in_val = train_speaker_set - val_speaker_set
    if missing_in_val:
        logger.warning(f"  ⚠ {len(missing_in_val)} speakers chỉ có trong train, "
                       f"không có val: {sorted(missing_in_val)[:10]}...")

    return train_records, val_records

# CORE: SAVE OUTPUTS
def save_filelists(
    train_records: List[Tuple[str, str, int]],
    val_records: List[Tuple[str, str, int]],
    config: ApplyFiltersConfig,
    logger: logging.Logger,
):
    """Lưu lại 2 filelist."""
    output_dir = Path(config.output_dir)
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list
    delim = config.delimiter

    def write_filelist(path: Path, records: List[Tuple[str, str, int]]):
        with open(path, "w", encoding="utf-8") as f:
            for wav_path, phoneme, sid in records:
                f.write(f"{wav_path}{delim}{phoneme}{delim}{sid}\n")

    write_filelist(train_path, train_records)
    write_filelist(val_path, val_records)

    logger.info(f"")
    logger.info(f"Đã ghi filelist:")
    logger.info(f"  Train : {train_path}  ({len(train_records):,} records)")
    logger.info(f"  Val   : {val_path}  ({len(val_records):,} records)")


def update_speaker_map(
    speaker_map_path: Path,
    counts_after: Dict[int, int],
    train_count: int,
    val_count: int,
    config: ApplyFiltersConfig,
    logger: logging.Logger,
):
    """
    Update speaker_id_map.json:
      - GIỮ NGUYÊN channel_to_speaker_id và speaker_id_to_channels
        (mapping này invariant với filter; speakers bị drop vẫn có entry,
         nhưng KHÔNG xuất hiện trong speaker_id_record_counts)
      - Cập nhật _metadata với min_samples/cap mới
      - Cập nhật speaker_id_record_counts theo final counts
    """
    with open(speaker_map_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Backup nếu chưa có
    backup_path = speaker_map_path.with_suffix(
        speaker_map_path.suffix + ".backup_before_filter"
    )
    if not backup_path.exists():
        shutil.copy2(speaker_map_path, backup_path)
        logger.info(f"  Backup created: {backup_path}")
    else:
        logger.info(f"  Backup đã tồn tại (giữ nguyên): {backup_path}")

    # Update metadata
    data["_metadata"]["min_samples_per_speaker"] = config.min_samples_per_speaker
    data["_metadata"]["cap_per_speaker"] = config.cap_per_speaker
    data["_metadata"]["drop_below_threshold"] = True  # step6b luôn drop hẳn
    data["_metadata"]["applied_by"] = "step6b_apply_filters.py"
    data["_metadata"]["unique_speaker_ids_after_filter"] = len(counts_after)
    data["_metadata"]["train_records"] = train_count
    data["_metadata"]["val_records"] = val_count

    # Update speaker_id_record_counts (chỉ giữ speakers còn lại sau filter)
    data["speaker_id_record_counts"] = {
        str(sid): counts_after[sid]
        for sid in sorted(counts_after.keys())
    }

    # GIỮ NGUYÊN channel_to_speaker_id + speaker_id_to_channels
    # (Lý do: mapping channel ↔ id là invariant. Speakers bị drop vẫn có
    #  trong mapping nhưng records của họ đã bị filter khỏi filelist.
    #  Giữ nguyên giúp ta có thể "undo" filter trong tương lai bằng cách
    #  re-run step6 cũ với data đầy đủ.)

    with open(speaker_map_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    logger.info(f"  Speaker map: {speaker_map_path}  (updated metadata + counts)")


def save_wavs_lists(
    train_records: List[Tuple[str, str, int]],
    val_records: List[Tuple[str, str, int]],
    original_train_path: Path,
    original_val_path: Path,
    original_train_records_count: int,
    original_val_records_count: int,
    config: ApplyFiltersConfig,
    logger: logging.Logger,
):
    """
    Tạo 2 file:
      - wavs_to_keep.txt: wav paths trong train+val cuối cùng (cần upload)
      - wavs_to_delete.txt: wav paths có trong filelist GỐC nhưng KHÔNG có
        trong filelist mới (có thể xóa khỏi disk)
    """
    output_dir = Path(config.output_dir)
    keep_path = output_dir / config.wavs_to_keep_file
    delete_path = output_dir / config.wavs_to_delete_file
    delim = config.delimiter

    # Wavs cuối cùng (giữ lại)
    kept_wavs = set()
    for records in (train_records, val_records):
        for wav_path, _, _ in records:
            kept_wavs.add(wav_path)

    # Wavs trong filelist GỐC
    all_original_wavs = set()
    for path in (original_train_path, original_val_path):
        if not path.exists():
            continue
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parsed = parse_record(line, delim)
                if parsed:
                    all_original_wavs.add(parsed[0])

    # KHẢ NĂNG đặc biệt: file train_list/val_list HIỆN TẠI đã bị overwrite trước
    # đó (nếu user chạy script 2 lần). Trong case đó, all_original_wavs ⊆ kept_wavs.
    # Để xử lý đúng: bypass `original_*_path` nếu sample count < kept count
    # (signaling rằng filelist đã bị overwrite).
    original_total = original_train_records_count + original_val_records_count
    if original_total < len(kept_wavs):
        # File GỐC đã không còn — không thể compute delete list chính xác
        logger.warning(
            f"  ⚠ filelist gốc đã bị overwrite ({original_total:,} records < "
            f"{len(kept_wavs):,} kept). KHÔNG ghi wavs_to_delete.txt."
        )
        deleted_wavs = set()
    else:
        deleted_wavs = all_original_wavs - kept_wavs

    # Sort để output deterministic
    kept_sorted = sorted(kept_wavs)
    deleted_sorted = sorted(deleted_wavs)

    with open(keep_path, "w", encoding="utf-8") as f:
        f.write(f"# wavs_to_keep.txt — Generated by step6b_apply_filters.py\n")
        f.write(f"# Tổng: {len(kept_sorted):,} wav files\n")
        f.write(f"# Train: {len(train_records):,}, Val: {len(val_records):,}\n")
        for wav in kept_sorted:
            f.write(f"{wav}\n")

    with open(delete_path, "w", encoding="utf-8") as f:
        f.write(f"# wavs_to_delete.txt — Generated by step6b_apply_filters.py\n")
        f.write(f"# Tổng: {len(deleted_sorted):,} wav files có thể XÓA khỏi disk\n")
        f.write(f"# Dùng delete_unused_wavs.py để xóa an toàn.\n")
        for wav in deleted_sorted:
            f.write(f"{wav}\n")

    logger.info(f"")
    logger.info(f"Đã ghi wavs lists:")
    logger.info(f"  Keep   : {keep_path}  ({len(kept_sorted):,} wavs cần giữ)")
    logger.info(f"  Delete : {delete_path}  ({len(deleted_sorted):,} wavs có thể xóa)")

# MAIN FLOW
def run(config: ApplyFiltersConfig, logger: logging.Logger):
    """Quy trình chính."""
    output_dir = Path(config.output_dir)
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list
    speaker_map_path = output_dir / config.speaker_map_file

    # === LOAD ===
    logger.info("=" * 60)
    logger.info("  Bước 6b.1: Đọc filelist gốc")
    logger.info("=" * 60)
    train_records = load_filelist(train_path, config.delimiter, logger)
    val_records = load_filelist(val_path, config.delimiter, logger)
    logger.info(f"  Tổng pool: {len(train_records) + len(val_records):,} records")

    # Lưu count GỐC để tính wavs_to_delete đúng (giúp detect overwrite case)
    original_train_count = len(train_records)
    original_val_count = len(val_records)

    if not speaker_map_path.exists():
        logger.error(f"Không tìm thấy speaker_id_map.json: {speaker_map_path}")
        raise FileNotFoundError(str(speaker_map_path))

    # Pool gộp
    pool = train_records + val_records

    # === FILTER + CAP ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6b.2: Filter + Cap")
    logger.info("=" * 60)
    # Dùng seed khác cho cap để tách biệt với split seed
    cap_seed = config.random_seed + 1
    filtered, counts_before, counts_after = filter_and_cap(
        pool,
        config.min_samples_per_speaker,
        config.cap_per_speaker,
        cap_seed,
        logger,
    )

    if not filtered:
        logger.error("Sau filter không còn record nào! Kiểm tra lại tham số.")
        return

    # === SHUFFLE + SPLIT ===
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6b.3: Shuffle + Split train/val")
    logger.info("=" * 60)
    train_new, val_new = shuffle_and_split(
        filtered,
        config.train_ratio,
        config.random_seed,
        logger,
    )

    # === SAVE OUTPUTS ===
    if config.dry_run:
        logger.info("")
        logger.info("=" * 60)
        logger.info("  [DRY-RUN] Bỏ qua bước ghi file")
        logger.info("=" * 60)
        logger.info(f"  Sẽ ghi (nếu không phải dry-run):")
        logger.info(f"    Train  : {len(train_new):,} records")
        logger.info(f"    Val    : {len(val_new):,} records")
        return

    logger.info("")
    logger.info("=" * 60)
    logger.info("  Bước 6b.4: Lưu outputs")
    logger.info("=" * 60)

    # Tạo wavs lists TRƯỚC khi overwrite filelist (để vẫn biết wavs gốc)
    save_wavs_lists(
        train_new, val_new,
        train_path, val_path,
        original_train_count, original_val_count,
        config, logger,
    )

    # Sau đó overwrite filelist
    save_filelists(train_new, val_new, config, logger)

    # Cuối cùng update speaker_id_map (sẽ tạo backup nếu chưa có)
    update_speaker_map(
        speaker_map_path,
        counts_after,
        len(train_new),
        len(val_new),
        config,
        logger,
    )

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 6b: Apply filters to existing filelists (no HF download)"
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
        help="Override min_samples_per_speaker (default: từ config = 1000)",
    )
    parser.add_argument(
        "--cap-per-speaker",
        type=int,
        default=None,
        help="Override cap_per_speaker (default: từ config = 20000; 0 = không cap)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log thống kê nhưng KHÔNG ghi file (để verify trước khi commit)",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = ApplyFiltersConfig.from_yaml(str(config_path))

    # Override từ CLI
    if args.min_samples is not None:
        config.min_samples_per_speaker = args.min_samples
    if args.cap_per_speaker is not None:
        config.cap_per_speaker = args.cap_per_speaker
    if args.dry_run:
        config.dry_run = True

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  BƯỚC 6B: APPLY FILTERS TO EXISTING FILELISTS")
    logger.info("=" * 60)
    logger.info(f"Config               : {config_path.resolve()}")
    logger.info(f"Min samples/speaker  : {config.min_samples_per_speaker:,}")
    logger.info(f"Cap per speaker      : {config.cap_per_speaker:,}  "
                f"({'ACTIVE' if config.cap_per_speaker > 0 else 'không cap'})")
    logger.info(f"Train ratio          : {config.train_ratio:.0%}")
    logger.info(f"Random seed          : {config.random_seed}")
    logger.info(f"Dry run              : {config.dry_run}")
    logger.info(f"Output dir           : {config.output_dir}")
    logger.info("")

    try:
        start = time.time()
        run(config, logger)
        elapsed = time.time() - start
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"  XONG! ({elapsed:.1f}s)")
        logger.info("=" * 60)
        if not config.dry_run:
            logger.info("")
            logger.info("Bước tiếp theo:")
            logger.info("  1. (Optional) Xem wavs_to_delete.txt để verify danh sách wav cần xóa")
            logger.info("  2. Chạy delete_unused_wavs.py để xóa wav không cần thiết:")
            logger.info("     python delete_unused_wavs.py")
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()