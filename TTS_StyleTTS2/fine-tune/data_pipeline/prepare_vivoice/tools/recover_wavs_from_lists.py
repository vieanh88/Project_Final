"""
=============================================================================
  RECOVER WAVS — Khôi phục các file .wav từ train/val list đã có
=============================================================================
Use case: Bạn ĐÃ chạy xong toàn bộ pipeline (step1 → step6) một lần và có:
            - vivoice_train_list.txt  (đã phonemize, đã có speaker_id)
            - vivoice_val_list.txt
          Nhưng vì lý do nào đó MẤT thư mục .wav (và có thể cả parquet cache).
          Script này KHÔI PHỤC chính xác các .wav cần thiết, KHÔNG chạy lại
          step3 (phonemize), step4 (vocab), step5/6 (filelist).

Chiến lược:
  1. Parse train + val lists → tập hợp NEEDED_INDICES (các vivoice_NNNNNNN cần)
  2. Tải lần lượt từng parquet shard từ HF Hub (rolling, có retry/resume)
  3. Iterate rows; chỉ decode + save .wav nếu global_index ∈ NEEDED_INDICES
  4. Sau shard đầu tiên hoàn tất: VERIFY alignment — kiểm tra các index dự kiến
     trong range shard đó có thực sự được tạo wav không. Sai → STOP ngay.
  5. Xóa parquet sau mỗi shard, update checkpoint sau mỗi shard.

Vì sao dùng được index từ list cũ:
  - Old step2 dùng `global_index += 1` mỗi sample (kể cả fail) → tên file
    = position trong dataset iteration order.
  - HF datasets iterate parquet shards theo alphabet, rows theo physical order.
  - pyarrow.iter_batches cũng vậy → 2 cách cho ra cùng thứ tự index (verify ở runtime).

Output:
  - output/vivoice_clean_wavs/vivoice_{N:07d}.wav  (giống pipeline cũ)
  - workdir/recover_checkpoint.json                 (resume state)
  - workdir/recover_missing.txt                     (index dự kiến nhưng không khôi phục được)

Chạy:
    python recover_wavs_from_lists.py
    python recover_wavs_from_lists.py --max-shards 1   # test alignment với 1 shard
=============================================================================
"""

import io
import os
import re
import sys
import json
import time
import shutil
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, List, Set, Tuple
from datetime import datetime

# Encoding fix Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# Tắt torchcodec
os.environ["DATASETS_AUDIO_DECODE_WITH_TORCHCODEC"] = "0"
os.environ["HF_DATASETS_AUDIO_DECODE_WITH_TORCHCODEC"] = "0"

import yaml
import numpy as np
import soundfile as sf
import pyarrow.parquet as pq
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub.utils import HfHubHTTPError, EntryNotFoundError


# =============================================================================
# CONFIG
# =============================================================================

@dataclass
class RecoverConfig:
    work_dir: str = "./workdir"
    output_dir: str = "./output"
    wav_subdir: str = "vivoice_clean_wavs"

    dataset_name: str = "capleaf/viVoice"
    dataset_split: Optional[str] = None

    target_sr: int = 24000
    bit_depth: int = 16

    # Train/val list paths
    train_list: str = "vivoice_train_list.txt"
    val_list: str = "vivoice_val_list.txt"
    list_delimiter: str = "|"

    # Tùy chọn
    skip_existing: bool = True
    hf_token: Optional[str] = None

    # Rolling parquet
    temp_parquet_subdir: str = "temp_parquet"
    checkpoint_file: str = "recover_checkpoint.json"
    download_max_retries: int = 5
    download_backoff_seconds: List[int] = field(
        default_factory=lambda: [10, 30, 60, 120, 300]
    )
    parquet_batch_size: int = 100

    # Validation: sau shard đầu tiên, % index expected phải khớp tối thiểu
    alignment_min_match_ratio: float = 0.99

    # Test mode
    max_shards: int = 0

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "RecoverConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step1 = full_config.get("step1_download", {})
        step2 = full_config.get("step2_extract", {})
        step5 = full_config.get("step5_filelist", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            dataset_name=step1.get("dataset_name", cls.dataset_name),
            dataset_split=step1.get("dataset_split", cls.dataset_split),
            wav_subdir=step2.get("wav_subdir", cls.wav_subdir),
            target_sr=step2.get("target_sr", cls.target_sr),
            bit_depth=step2.get("bit_depth", cls.bit_depth),
            train_list=step5.get("train_list", cls.train_list),
            val_list=step5.get("val_list", cls.val_list),
            list_delimiter=step5.get("delimiter", cls.list_delimiter),
            skip_existing=step2.get("skip_existing", cls.skip_existing),
        )


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "recover_wavs.log"
    root = logging.getLogger()
    for h in root.handlers[:]:
        root.removeHandler(h)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_file), encoding="utf-8", mode="a"),
        ],
    )
    return logging.getLogger("recover_wavs")


# =============================================================================
# AUDIO UTILS
# =============================================================================

def to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=0)


def resample_audio(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return audio
    import librosa
    return librosa.resample(audio.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    peak = np.abs(audio).max()
    if peak > 0:
        return audio / peak
    return audio


def save_wav(audio: np.ndarray, path: Path, sr: int, bit_depth: int = 16):
    subtype_map = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}
    sf.write(str(path), audio, sr, subtype=subtype_map.get(bit_depth, "PCM_16"))


def decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    try:
        audio, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
        return audio, sr
    except Exception as e_sf:
        try:
            import librosa
            audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=None, mono=False)
            return audio.astype(np.float32), sr
        except Exception as e_lib:
            raise ValueError(f"sf={e_sf} | librosa={e_lib}")


def decode_and_save(
    audio_bytes: bytes,
    wav_path: Path,
    target_sr: int,
    bit_depth: int,
) -> Tuple[bool, Optional[str]]:
    """
    Decode audio bytes → save wav. Trả về (success, error_msg).
    Logic giống hệt step2_extract_audio.process_single_sample (chỉ phần audio).
    """
    try:
        if not audio_bytes:
            return False, "Audio bytes rỗng"
        audio, orig_sr = decode_audio_bytes(audio_bytes)
        audio = to_mono(audio)
        audio = resample_audio(audio, orig_sr, target_sr)
        if len(audio) / target_sr < 0.1:
            return False, f"Audio quá ngắn: {len(audio)/target_sr:.3f}s"
        if np.all(audio == 0):
            return False, "Audio toàn silence"
        audio = normalize_audio(audio)
        save_wav(audio, wav_path, target_sr, bit_depth)
        return True, None
    except Exception as e:
        return False, str(e)


# =============================================================================
# PARSE TRAIN/VAL LISTS
# =============================================================================

# Pattern khớp 'vivoice_NNNNNNN.wav' (7 chữ số, có thể nằm sau backslash hay slash)
WAV_INDEX_PATTERN = re.compile(r"vivoice_(\d{7})\.wav", re.IGNORECASE)


def parse_list_file(path: Path, delimiter: str, logger: logging.Logger) -> Set[int]:
    """Parse 1 file train/val list → set các integer index."""
    if not path.exists():
        logger.error(f"Không tìm thấy file list: {path}")
        return set()

    indices: Set[int] = set()
    bad_lines = 0
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            # Cột đầu tiên là wav_path
            wav_part = line.split(delimiter)[0]
            m = WAV_INDEX_PATTERN.search(wav_part)
            if m:
                indices.add(int(m.group(1)))
            else:
                bad_lines += 1
                if bad_lines <= 3:
                    logger.warning(f"  {path.name}:{line_no}: không parse được wav index từ '{wav_part[:80]}'")

    if bad_lines > 0:
        logger.warning(f"  {path.name}: bỏ qua {bad_lines} dòng không parse được index")
    logger.info(f"  {path.name}: {len(indices):,} indices unique")
    return indices


def load_needed_indices(config: RecoverConfig, logger: logging.Logger) -> Set[int]:
    """Đọc cả train + val list, gộp thành 1 set."""
    work_dir = Path(config.work_dir)

    # Train/val list thường nằm trong work_dir hoặc output_dir, tùy step5/6
    # config. Thử cả 2 location.
    candidates_train = [
        work_dir / config.train_list,
        Path(config.output_dir) / config.train_list,
        Path(config.train_list),
    ]
    candidates_val = [
        work_dir / config.val_list,
        Path(config.output_dir) / config.val_list,
        Path(config.val_list),
    ]

    train_path = next((p for p in candidates_train if p.exists()), None)
    val_path = next((p for p in candidates_val if p.exists()), None)

    if train_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy {config.train_list} ở: {[str(p) for p in candidates_train]}"
        )
    if val_path is None:
        raise FileNotFoundError(
            f"Không tìm thấy {config.val_list} ở: {[str(p) for p in candidates_val]}"
        )

    logger.info(f"Train list : {train_path}")
    logger.info(f"Val list   : {val_path}")

    train_idx = parse_list_file(train_path, config.list_delimiter, logger)
    val_idx = parse_list_file(val_path, config.list_delimiter, logger)

    # Kiểm tra overlap (không nên có)
    overlap = train_idx & val_idx
    if overlap:
        logger.warning(f"  Có {len(overlap)} index xuất hiện ở CẢ train và val (bất thường)")

    needed = train_idx | val_idx
    logger.info(f"Tổng cần khôi phục: {len(needed):,} wav files")
    if needed:
        logger.info(f"  Index range : {min(needed)} → {max(needed)}")

    return needed


# =============================================================================
# CHECKPOINT
# =============================================================================

@dataclass
class Checkpoint:
    version: int = 1
    dataset_name: str = ""
    completed_shards: List[str] = field(default_factory=list)
    global_index: int = 0           # vị trí iteration tiếp theo (qua TẤT CẢ samples)
    total_recovered: int = 0        # số wav đã khôi phục được
    total_decode_errors: int = 0    # số wav cần khôi phục nhưng decode fail
    last_update: str = ""

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "dataset_name": self.dataset_name,
            "completed_shards": self.completed_shards,
            "global_index": self.global_index,
            "total_recovered": self.total_recovered,
            "total_decode_errors": self.total_decode_errors,
            "last_update": self.last_update,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Checkpoint":
        return cls(
            version=data.get("version", 1),
            dataset_name=data.get("dataset_name", ""),
            completed_shards=data.get("completed_shards", []),
            global_index=data.get("global_index", 0),
            total_recovered=data.get("total_recovered", 0),
            total_decode_errors=data.get("total_decode_errors", 0),
            last_update=data.get("last_update", ""),
        )


def load_checkpoint(path: Path, dataset_name: str, logger: logging.Logger) -> Checkpoint:
    if not path.exists():
        return Checkpoint(dataset_name=dataset_name)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        ckpt = Checkpoint.from_dict(data)
        if ckpt.dataset_name and ckpt.dataset_name != dataset_name:
            logger.warning(f"Checkpoint cũ dùng '{ckpt.dataset_name}' ≠ '{dataset_name}' → reset")
            return Checkpoint(dataset_name=dataset_name)
        logger.info(
            f"Resume từ checkpoint: {len(ckpt.completed_shards)} shard xong, "
            f"global_index={ckpt.global_index:,}, recovered={ckpt.total_recovered:,}"
        )
        return ckpt
    except Exception as e:
        logger.warning(f"Lỗi đọc checkpoint ({e}) → reset")
        return Checkpoint(dataset_name=dataset_name)


def save_checkpoint(ckpt: Checkpoint, path: Path):
    ckpt.last_update = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(ckpt.to_dict(), f, ensure_ascii=False, indent=2)
    os.replace(str(tmp), str(path))


# =============================================================================
# DOWNLOAD WITH RETRY
# =============================================================================

def list_parquet_shards(
    dataset_name: str,
    token: Optional[str],
    dataset_split: Optional[str],
    logger: logging.Logger,
) -> List[str]:
    api = HfApi()
    files = api.list_repo_files(repo_id=dataset_name, repo_type="dataset", token=token)
    parquet = sorted([f for f in files if f.endswith(".parquet")])
    if dataset_split:
        filt = [f for f in parquet if dataset_split in f]
        if filt:
            parquet = filt
            logger.info(f"Filter split='{dataset_split}': {len(parquet)} files")
    return parquet


def download_shard_with_retry(
    dataset_name: str,
    shard_name: str,
    local_dir: Path,
    token: Optional[str],
    max_retries: int,
    backoff: List[int],
    logger: logging.Logger,
) -> Path:
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            p = hf_hub_download(
                repo_id=dataset_name,
                filename=shard_name,
                repo_type="dataset",
                local_dir=str(local_dir),
                token=token,
            )
            return Path(p)
        except (HfHubHTTPError, EntryNotFoundError, ConnectionError, OSError, TimeoutError) as e:
            last_err = e
            if attempt >= max_retries:
                break
            wait = backoff[min(attempt - 1, len(backoff) - 1)]
            logger.warning(f"  Tải {shard_name} fail (lần {attempt}/{max_retries}): {e}. Đợi {wait}s")
            time.sleep(wait)
        except Exception as e:
            logger.error(f"  Lỗi không retry được: {e}")
            raise
    raise RuntimeError(f"Không tải được {shard_name} sau {max_retries} lần. Last err: {last_err}")


# =============================================================================
# PROCESS SHARD
# =============================================================================

def process_shard(
    parquet_path: Path,
    start_index: int,
    needed_indices: Set[int],
    wav_dir: Path,
    config: RecoverConfig,
    logger: logging.Logger,
) -> dict:
    """
    Xử lý 1 shard: iterate rows, decode + save chỉ khi index ∈ needed_indices.
    Returns: stats dict {"rows_in_shard", "needed_in_shard", "recovered",
                          "decode_errors", "skip_existing", "error_details"}
    """
    from tqdm import tqdm

    pf = pq.ParquetFile(str(parquet_path))
    total_rows = pf.metadata.num_rows
    end_index_exclusive = start_index + total_rows

    # Tìm các index cần khôi phục thuộc range [start_index, end_index_exclusive)
    needed_in_shard = {i for i in needed_indices if start_index <= i < end_index_exclusive}

    schema_names = pf.schema_arrow.names
    if "audio" not in schema_names:
        raise ValueError(f"Parquet thiếu cột 'audio'. Cột có: {schema_names}")

    stats = {
        "rows_in_shard": total_rows,
        "needed_in_shard": len(needed_in_shard),
        "recovered": 0,
        "decode_errors": 0,
        "skip_existing": 0,
        "error_details": {},
    }

    current_index = start_index
    pbar = tqdm(total=total_rows, desc=f"  {parquet_path.name}", ncols=100)

    try:
        for batch in pf.iter_batches(batch_size=config.parquet_batch_size, columns=["audio"]):
            df = batch.to_pandas()
            for _, row in df.iterrows():
                if current_index in needed_in_shard:
                    wav_path = wav_dir / f"vivoice_{current_index:07d}.wav"

                    # Skip nếu wav đã tồn tại (resume an toàn)
                    if config.skip_existing and wav_path.exists():
                        stats["skip_existing"] += 1
                        stats["recovered"] += 1  # coi như đã có
                    else:
                        # Decode + save
                        audio_field = row["audio"]
                        audio_bytes = None
                        if isinstance(audio_field, dict):
                            audio_bytes = audio_field.get("bytes")
                        elif isinstance(audio_field, bytes):
                            audio_bytes = audio_field

                        ok, err = decode_and_save(
                            audio_bytes=audio_bytes,
                            wav_path=wav_path,
                            target_sr=config.target_sr,
                            bit_depth=config.bit_depth,
                        )
                        if ok:
                            stats["recovered"] += 1
                        else:
                            stats["decode_errors"] += 1
                            err = err or "Unknown"
                            stats["error_details"][err] = stats["error_details"].get(err, 0) + 1
                # else: index không cần → skip decode hoàn toàn (chỉ tăng counter)

                current_index += 1
                pbar.update(1)
    finally:
        pbar.close()

    return stats


# =============================================================================
# ALIGNMENT VERIFICATION
# =============================================================================

def verify_alignment(
    wav_dir: Path,
    needed_indices_in_shard: Set[int],
    shard_stats: dict,
    config: RecoverConfig,
    logger: logging.Logger,
) -> bool:
    """
    Kiểm tra alignment sau shard đầu tiên: các index dự kiến khôi phục thuộc
    shard này có thực sự được tạo wav không.

    Returns True nếu OK, False nếu sai lệch nhiều → user nên dừng.
    """
    if not needed_indices_in_shard:
        logger.warning(
            "  Không có index nào thuộc shard này — không thể verify alignment. "
            "Nếu shard đầu tiên RỖNG mong đợi, có thể bỏ qua. Nếu không → kiểm tra lại."
        )
        return True

    # Kiểm tra xem các wav file dự kiến có thực sự tồn tại không
    actually_exists = sum(
        1 for idx in needed_indices_in_shard
        if (wav_dir / f"vivoice_{idx:07d}.wav").exists()
    )
    expected = len(needed_indices_in_shard)
    match_ratio = actually_exists / expected if expected > 0 else 0.0

    logger.info("")
    logger.info("=" * 60)
    logger.info("  KIỂM TRA ALIGNMENT (sau shard đầu tiên)")
    logger.info("=" * 60)
    logger.info(f"  Index dự kiến trong shard này  : {expected:,}")
    logger.info(f"  Wav thực sự được tạo           : {actually_exists:,}")
    logger.info(f"  Tỷ lệ khớp                     : {match_ratio*100:.2f}%")
    logger.info(f"  Decode errors                  : {shard_stats['decode_errors']:,}")
    logger.info(f"  Threshold tối thiểu            : {config.alignment_min_match_ratio*100:.0f}%")

    if match_ratio < config.alignment_min_match_ratio:
        logger.error("")
        logger.error("ALIGNMENT KHÔNG ĐỦ! Có thể do:")
        logger.error("  1. Thứ tự iterate parquet shard khác lần đầu")
        logger.error("  2. Nội dung dataset trên HF đã đổi từ lần đầu bạn chạy")
        logger.error("  3. Nhiều sample bị decode error mới")
        logger.error("Đề xuất: kiểm tra log decode errors phía trên + dừng pipeline.")
        return False

    logger.info("  → OK, alignment khớp. Tiếp tục các shard sau.")
    return True


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_recovery(config: RecoverConfig, logger: logging.Logger):
    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_dir = output_dir / config.wav_subdir
    wav_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = work_dir / config.temp_parquet_subdir
    temp_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = work_dir / config.checkpoint_file
    missing_path = work_dir / "recover_missing.txt"

    # --- Parse train/val ---
    logger.info("Đọc train/val list để xác định wav cần khôi phục...")
    needed = load_needed_indices(config, logger)
    if not needed:
        logger.error("Không có index nào cần khôi phục → dừng.")
        return

    # --- Checkpoint ---
    ckpt = load_checkpoint(checkpoint_path, config.dataset_name, logger)
    completed_set: Set[str] = set(ckpt.completed_shards)

    # --- List shards ---
    logger.info("Liệt kê parquet shards trên HF...")
    all_shards = list_parquet_shards(config.dataset_name, config.hf_token, config.dataset_split, logger)
    if not all_shards:
        logger.error("Không có parquet nào!")
        return
    logger.info(f"Tổng: {len(all_shards)} shards. Đã xong: {len(completed_set)}.")

    pending = [s for s in all_shards if s not in completed_set]
    if config.max_shards > 0:
        pending = pending[: config.max_shards]
        logger.info(f"Test mode: chỉ chạy {len(pending)} shards")

    if not pending:
        logger.info("Tất cả shards đã xong!")
        _write_missing_report(needed, wav_dir, missing_path, logger)
        return

    pipeline_start = time.time()
    is_first_shard = (len(completed_set) == 0)

    for shard_idx, shard_name in enumerate(pending, start=1):
        logger.info("")
        logger.info(f"[{shard_idx}/{len(pending)}] Shard: {shard_name}")
        shard_start = time.time()

        # --- Tải ---
        try:
            local_path = download_shard_with_retry(
                dataset_name=config.dataset_name,
                shard_name=shard_name,
                local_dir=temp_dir,
                token=config.hf_token,
                max_retries=config.download_max_retries,
                backoff=config.download_backoff_seconds,
                logger=logger,
            )
            size_mb = local_path.stat().st_size / (1024 * 1024)
            logger.info(f"  Đã tải: {size_mb:.1f} MB")
        except Exception as e:
            logger.error(f"  Bỏ qua shard này (download fail): {e}")
            continue

        # --- Tính range index của shard này ĐỂ DÙNG cho verify alignment ---
        # global_index hiện tại = start của shard này
        shard_start_index = ckpt.global_index

        # --- Process ---
        try:
            shard_stats = process_shard(
                parquet_path=local_path,
                start_index=ckpt.global_index,
                needed_indices=needed,
                wav_dir=wav_dir,
                config=config,
                logger=logger,
            )
        except Exception as e:
            logger.exception(f"  Lỗi xử lý shard: {e}")
            try:
                local_path.unlink()
            except Exception:
                pass
            continue

        # --- Verify alignment sau shard đầu tiên ---
        if is_first_shard:
            shard_end_idx = shard_start_index + shard_stats["rows_in_shard"]
            needed_in_first_shard = {i for i in needed if shard_start_index <= i < shard_end_idx}
            ok = verify_alignment(wav_dir, needed_in_first_shard, shard_stats, config, logger)
            if not ok:
                logger.error("DỪNG pipeline để bạn kiểm tra. Checkpoint chưa save shard này.")
                # Cleanup parquet
                try:
                    local_path.unlink()
                except Exception:
                    pass
                sys.exit(2)
            is_first_shard = False  # đã verify xong

        # --- Update checkpoint ---
        ckpt.global_index += shard_stats["rows_in_shard"]
        ckpt.total_recovered += shard_stats["recovered"]
        ckpt.total_decode_errors += shard_stats["decode_errors"]
        ckpt.completed_shards.append(shard_name)
        save_checkpoint(ckpt, checkpoint_path)

        # --- Xóa parquet ---
        try:
            local_path.unlink()
            logger.info(f"  Đã xóa parquet → free {size_mb:.1f} MB")
        except Exception as e:
            logger.warning(f"  Không xóa được parquet: {e}")

        elapsed = time.time() - shard_start
        logger.info(
            f"  Shard xong: needed={shard_stats['needed_in_shard']:,}, "
            f"recovered={shard_stats['recovered']:,} "
            f"(skip_existing={shard_stats['skip_existing']:,}), "
            f"errors={shard_stats['decode_errors']:,}, time={elapsed:.1f}s"
        )
        if shard_stats["error_details"]:
            top = sorted(shard_stats["error_details"].items(), key=lambda x: -x[1])[:3]
            logger.info(f"  Top errors: {', '.join(f'{k}({v})' for k, v in top)}")

    # --- Cleanup temp dir ---
    try:
        if not list(temp_dir.rglob("*")):
            shutil.rmtree(temp_dir, ignore_errors=True)
    except Exception:
        pass

    pipeline_elapsed = time.time() - pipeline_start

    # --- Final report ---
    _write_missing_report(needed, wav_dir, missing_path, logger)

    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ KHÔI PHỤC")
    logger.info("=" * 60)
    logger.info(f"  Thời gian       : {pipeline_elapsed:.1f}s ({pipeline_elapsed/60:.1f} phút)")
    logger.info(f"  Shard hoàn thành: {len(ckpt.completed_shards)}/{len(all_shards)}")
    logger.info(f"  Cần khôi phục   : {len(needed):,}")
    logger.info(f"  Đã khôi phục    : {ckpt.total_recovered:,}")
    logger.info(f"  Decode errors   : {ckpt.total_decode_errors:,}")
    rate = ckpt.total_recovered / len(needed) * 100 if needed else 0
    logger.info(f"  Tỷ lệ           : {rate:.2f}%")

def _write_missing_report(
    needed: Set[int],
    wav_dir: Path,
    missing_path: Path,
    logger: logging.Logger,
):
    """Liệt kê các index cần nhưng wav chưa tồn tại."""
    missing = sorted(idx for idx in needed if not (wav_dir / f"vivoice_{idx:07d}.wav").exists())
    if missing:
        with open(missing_path, "w", encoding="utf-8") as f:
            f.write(f"# Index thiếu wav (tổng {len(missing):,})\n")
            f.write(f"# Generated: {datetime.now()}\n")
            for idx in missing:
                f.write(f"vivoice_{idx:07d}\n")
        logger.warning(f"  Còn {len(missing):,} wav chưa khôi phục được → {missing_path}")
    else:
        logger.info(f"  ✓ Tất cả wav cần thiết đã khôi phục đầy đủ!")
        if missing_path.exists():
            try:
                missing_path.unlink()
            except Exception:
                pass

# MAIN
def main():
    parser = argparse.ArgumentParser(description="Khôi phục wav từ train/val list đã có")
    parser.add_argument("--config", "-c", default="config.yaml")
    parser.add_argument("--max-shards", type=int, default=0,
                        help="Giới hạn số shard (0 = tất cả). Dùng --max-shards 1 để test alignment.")
    args = parser.parse_args()

    config_path = Path(args.config) # Mặc định tìm config.yaml ở current dir, hoặc có thể chỉ định path khác

    # Load .env (Đã sửa lỗi path resolution)
    abs_config_dir = config_path.resolve().parent
    for env_path in [
        abs_config_dir / ".env",
        abs_config_dir.parent / ".env",
        abs_config_dir.parent.parent / ".env",
        abs_config_dir.parent.parent.parent / ".env",
    ]:
        if env_path.exists():
            load_dotenv(str(env_path), override=True)
            break

    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy: {config_path}")
        sys.exit(1)

    config = RecoverConfig.from_yaml(str(config_path))
    if args.max_shards > 0:
        config.max_shards = args.max_shards

    config.hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")

    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    logger.info("=" * 60)
    logger.info("  RECOVER WAVS — Khôi phục wav từ train/val list đã có")
    logger.info("=" * 60)
    logger.info(f"Dataset    : {config.dataset_name}")
    logger.info(f"Wav output : {Path(config.output_dir) / config.wav_subdir}")
    logger.info(f"HF Token   : {'Có' if config.hf_token else 'KHÔNG'}")

    if not config.hf_token:
        logger.warning("Thiếu HF_TOKEN — capleaf/viVoice là gated dataset, sẽ fail.")

    try:
        run_recovery(config, logger)
        logger.info("")
        logger.info("Xong! Bạn có thể train StyleTTS2 trực tiếp với train/val list cũ.")
        logger.info("(Không cần chạy step3, step4, step5, step6 nữa.)")
    except KeyboardInterrupt:
        logger.warning("\nĐã dừng (Ctrl+C). Checkpoint đã save — chạy lại để resume.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        logger.info("Checkpoint đã save. Chạy lại để tiếp tục.")
        sys.exit(1)

if __name__ == "__main__":
    main()