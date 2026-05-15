"""
=============================================================================
  DELETE UNUSED WAVS — Xóa các file .wav không còn cần thiết
=============================================================================
Mục tiêu: Xóa các file wav được liệt kê trong `output/wavs_to_delete.txt`
          (output của step6b_apply_filters.py) để giải phóng disk.

Use case:
  - Đã chạy step6b_apply_filters.py xong
  - File output/wavs_to_delete.txt chứa ~192,661 wav paths cần xóa
    (sau filter min_samples=1000 + cap=20000 ⇒ giải phóng ~35.5 GB)

SAFETY MECHANISMS (4 LỚP BẢO VỆ):
  Lớp 1: Verify input file tồn tại + có dòng
  Lớp 2: CROSS-CHECK với vivoice_train_list.txt + vivoice_val_list.txt
         → Đảm bảo KHÔNG xóa nhầm wav vẫn còn trong filelist mới.
         → Nếu có overlap → ABORT ngay (lỗi nghiêm trọng).
  Lớp 3: Confirm prompt "gõ DELETE" để xác nhận (bypass bằng --yes)
  Lớp 4: --dry-run chỉ log + không xóa thật

EDGE CASES được handle:
  - File đã bị xóa từ trước (chạy lần 2) → count "already_missing", không lỗi
  - Permission error (file đang bị process khác giữ) → log + continue
  - Trùng đường dẫn trong wavs_to_delete.txt → dedupe
  - Comment lines (bắt đầu '#') → skip
  - Đường dẫn Windows vs Linux → cảnh báo nếu detect mismatch

Đầu vào : - output/wavs_to_delete.txt   (từ step6b_apply_filters.py)
           - output/vivoice_train_list.txt  (CROSS-CHECK)
           - output/vivoice_val_list.txt    (CROSS-CHECK)

Đầu ra  : - File wav bị xóa khỏi disk (~35 GB free)
           - workdir/logs/delete_unused_wavs.log
           - workdir/logs/delete_unused_wavs_failed.txt
             (chỉ tạo nếu có file fail, để retry)

Chạy lệnh:
    python delete_unused_wavs.py                    # interactive (prompt)
    python delete_unused_wavs.py --dry-run          # log + KHÔNG xóa
    python delete_unused_wavs.py --yes              # skip prompt (auto)
    python delete_unused_wavs.py --config config.yaml
=============================================================================
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List, Set, Tuple
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
class DeleteConfig:
    """Cấu hình cho delete_unused_wavs."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"

    # File input (đều trong output_dir)
    wavs_to_delete_file: str = "wavs_to_delete.txt"
    train_list: str = "vivoice_train_list.txt"
    val_list: str = "vivoice_val_list.txt"

    # Delimiter để parse filelist (lấy wav_path)
    delimiter: str = "|"

    # Modes
    dry_run: bool = False
    skip_confirm: bool = False

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DeleteConfig":
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
            delimiter=step6.get("delimiter", step5.get("delimiter", cls.delimiter)),
        )


# =============================================================================
# LOGGING
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "delete_unused_wavs.log"

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
    return logging.getLogger("delete_unused_wavs")


# =============================================================================
# CORE: LOAD WAVS_TO_DELETE
# =============================================================================

def load_wavs_to_delete(path: Path, logger: logging.Logger) -> List[str]:
    """
    Đọc wavs_to_delete.txt, return list các wav paths (đã dedupe).

    Skip:
      - Dòng trống
      - Comment (bắt đầu '#')
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}\n"
            f"Hãy chạy step6b_apply_filters.py trước để tạo file này!"
        )

    wavs = []
    seen: Set[str] = set()
    duplicates = 0
    comment_count = 0
    empty_count = 0

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                empty_count += 1
                continue
            if line.startswith("#"):
                comment_count += 1
                continue
            if line in seen:
                duplicates += 1
                continue
            seen.add(line)
            wavs.append(line)

    logger.info(f"")
    logger.info(f"Đọc {path.name}:")
    logger.info(f"  Wavs cần xóa (unique): {len(wavs):,}")
    if duplicates:
        logger.info(f"  Trùng lặp (skip)     : {duplicates:,}")
    if comment_count:
        logger.info(f"  Comment lines        : {comment_count:,}")
    if empty_count:
        logger.info(f"  Empty lines          : {empty_count:,}")

    return wavs


# =============================================================================
# CORE: LOAD WAVS_TO_KEEP (CROSS-CHECK)
# =============================================================================

def load_wavs_in_filelists(
    train_path: Path,
    val_path: Path,
    delimiter: str,
    logger: logging.Logger,
) -> Set[str]:
    """
    Đọc 2 filelist (train + val mới sau filter), trích cột wav_path.

    Return set các wav paths đang ĐƯỢC SỬ DỤNG (cần GIỮ, không xóa).
    """
    kept_wavs: Set[str] = set()
    for path in (train_path, val_path):
        if not path.exists():
            logger.warning(f"  ⚠ Không tìm thấy {path} — bỏ qua")
            continue
        n = 0
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                # Lấy wav_path = phần trước delimiter ĐẦU TIÊN
                parts = line.split(delimiter, 1)
                if not parts:
                    continue
                kept_wavs.add(parts[0])
                n += 1
        logger.info(f"  {path.name}: {n:,} wavs")

    logger.info(f"  Tổng wavs GIỮ (unique): {len(kept_wavs):,}")
    return kept_wavs


# =============================================================================
# CORE: SAFETY CROSS-CHECK
# =============================================================================

def cross_check_overlap(
    wavs_to_delete: List[str],
    wavs_to_keep: Set[str],
    logger: logging.Logger,
) -> List[str]:
    """
    LỚP 2 SAFETY: verify không có wav nào trong wavs_to_delete xuất hiện
    cả trong wavs_to_keep (= filelist mới sau filter).

    Nếu có overlap → có nghĩa wavs_to_delete.txt đã LỖI THỜI (chạy step6b sau
    khi sinh wavs_to_delete.txt nên file đó stale) hoặc bug logic.
    → ABORT để tránh xóa nhầm.

    Return: list các wav SẠCH (không trong wavs_to_keep) — tức sẵn sàng xóa.
    """
    overlap = [w for w in wavs_to_delete if w in wavs_to_keep]
    safe_to_delete = [w for w in wavs_to_delete if w not in wavs_to_keep]

    if overlap:
        logger.error(f"")
        logger.error(f"╔══════════════════════════════════════════════════════════╗")
        logger.error(f"║  LỖI NGHIÊM TRỌNG: CROSS-CHECK FAILED                    ║")
        logger.error(f"╚══════════════════════════════════════════════════════════╝")
        logger.error(f"")
        logger.error(f"  Phát hiện {len(overlap):,} wavs xuất hiện TRONG CẢ:")
        logger.error(f"    - wavs_to_delete.txt (đáng lẽ phải xóa)")
        logger.error(f"    - vivoice_train_list.txt / vivoice_val_list.txt (đang dùng)")
        logger.error(f"")
        logger.error(f"  → wavs_to_delete.txt có thể đã LỖI THỜI")
        logger.error(f"    (bạn đã chạy step6b với tham số khác lần trước?)")
        logger.error(f"")
        logger.error(f"  CÁCH SỬA:")
        logger.error(f"    1. Chạy lại step6b_apply_filters.py để regenerate wavs_to_delete.txt")
        logger.error(f"    2. Sau đó chạy lại delete_unused_wavs.py")
        logger.error(f"")
        logger.error(f"  Ví dụ wavs bị overlap (max 5):")
        for w in overlap[:5]:
            logger.error(f"    - {w}")
        return []  # signal abort

    logger.info(f"  ✓ Cross-check OK: KHÔNG có wav nào overlap giữa delete và keep lists")
    return safe_to_delete


# =============================================================================
# CORE: PATH SANITY CHECK
# =============================================================================

def check_path_compatibility(
    wavs: List[str],
    logger: logging.Logger,
) -> bool:
    """
    Kiểm tra paths trong list có tương thích với OS hiện tại không.

    Ví dụ: chạy script trên Linux với paths Windows "D:/..." → fail
    """
    if not wavs:
        return True

    sample = wavs[0]
    current_os_is_windows = (sys.platform == "win32")
    looks_like_windows_path = (
        len(sample) >= 3 and sample[1] == ":" and sample[2] in ("/", "\\")
    )

    if looks_like_windows_path and not current_os_is_windows:
        logger.error(f"")
        logger.error(f"  ⚠ PATH MISMATCH: paths trong wavs_to_delete.txt có vẻ là")
        logger.error(f"    Windows path (\"{sample[:30]}...\") nhưng script đang")
        logger.error(f"    chạy trên {sys.platform}.")
        logger.error(f"")
        logger.error(f"    Hãy chạy script này trên cùng OS với máy đã sinh")
        logger.error(f"    wavs_to_delete.txt (thường là Windows local).")
        return False

    return True


# =============================================================================
# CORE: SIZE ESTIMATION
# =============================================================================

def estimate_total_size(wavs: List[str], logger: logging.Logger) -> int:
    """
    Đo tổng dung lượng các wav sẽ xóa (sample 1000 file để ước lượng nếu list lớn).

    Return: tổng bytes
    """
    if not wavs:
        return 0

    # Nếu < 5000 files → đo chính xác
    # Nếu >= 5000 files → sample 1000 file để ước lượng
    SAMPLE_THRESHOLD = 5000
    SAMPLE_SIZE = 1000

    total_bytes = 0
    counted = 0
    missing = 0

    if len(wavs) < SAMPLE_THRESHOLD:
        # Đo chính xác
        logger.info(f"  Đo chính xác dung lượng ({len(wavs):,} files)...")
        for w in wavs:
            try:
                total_bytes += os.path.getsize(w)
                counted += 1
            except (OSError, FileNotFoundError):
                missing += 1
        if missing:
            logger.info(f"  ({missing:,} files đã không tồn tại — đã từng xóa?)")
        return total_bytes

    # Sample-based estimation
    import random
    rng = random.Random(42)
    sample = rng.sample(wavs, SAMPLE_SIZE)

    sample_bytes = 0
    sample_counted = 0
    sample_missing = 0
    for w in sample:
        try:
            sample_bytes += os.path.getsize(w)
            sample_counted += 1
        except (OSError, FileNotFoundError):
            sample_missing += 1

    if sample_counted == 0:
        logger.warning(f"  Không đo được dung lượng (tất cả files đã không tồn tại?)")
        return 0

    avg_bytes = sample_bytes / sample_counted
    # Áp dụng tỉ lệ existing cho toàn list
    existing_ratio = sample_counted / SAMPLE_SIZE
    estimated_existing = int(len(wavs) * existing_ratio)
    total_bytes = int(avg_bytes * estimated_existing)

    logger.info(f"  Ước lượng dung lượng (sample {SAMPLE_SIZE}/{len(wavs):,}):")
    logger.info(f"    Avg size/file       : {avg_bytes / 1024:.1f} KB")
    logger.info(f"    Existing rate       : {existing_ratio * 100:.1f}%")
    logger.info(f"    Estimated existing  : {estimated_existing:,} files")
    logger.info(f"    Estimated total size: ~{total_bytes / 1e9:.1f} GB")

    return total_bytes


def format_size(bytes_val: int) -> str:
    """Format bytes thành KB/MB/GB cho dễ đọc."""
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if bytes_val < 1024:
            return f"{bytes_val:.1f} {unit}"
        bytes_val /= 1024
    return f"{bytes_val:.1f} PB"


# =============================================================================
# CORE: CONFIRM PROMPT
# =============================================================================

def confirm_deletion(
    n_files: int,
    estimated_bytes: int,
    output_dir: Path,
    logger: logging.Logger,
) -> bool:
    """
    LỚP 3 SAFETY: yêu cầu user gõ 'DELETE' để confirm.

    Return: True nếu user confirm, False nếu cancel.
    """
    logger.info(f"")
    logger.info(f"╔══════════════════════════════════════════════════════════╗")
    logger.info(f"║  ACTION CONFIRMATION REQUIRED                            ║")
    logger.info(f"╚══════════════════════════════════════════════════════════╝")
    logger.info(f"")
    logger.info(f"  Số file sẽ xóa     : {n_files:,}")
    logger.info(f"  Ước lượng dung lượng: {format_size(estimated_bytes)}")
    logger.info(f"  Output dir         : {output_dir}")
    logger.info(f"")
    logger.info(f"  ⚠ HÀNH ĐỘNG NÀY KHÔNG THỂ HOÀN TÁC.")
    logger.info(f"    Để khôi phục các file đã xóa, bạn sẽ phải tải lại")
    logger.info(f"    từ HuggingFace (~5h trên 90 Mbps).")
    logger.info(f"")
    logger.info(f"  Gõ chữ 'DELETE' (uppercase) để xác nhận, hoặc bất cứ thứ")
    logger.info(f"  gì khác để hủy bỏ:")

    try:
        # Đọc thẳng từ stdin để tránh logger flush issues
        sys.stdout.write("  > ")
        sys.stdout.flush()
        response = input().strip()
    except (EOFError, KeyboardInterrupt):
        logger.info(f"")
        logger.info(f"  → Hủy bỏ (no input)")
        return False

    if response == "DELETE":
        logger.info(f"  ✓ Đã xác nhận. Bắt đầu xóa...")
        return True
    else:
        logger.info(f"  → Hủy bỏ (gõ '{response}' thay vì 'DELETE')")
        return False


# =============================================================================
# CORE: ACTUAL DELETION
# =============================================================================

def delete_files(
    wavs: List[str],
    dry_run: bool,
    logger: logging.Logger,
) -> dict:
    """
    Xóa từng wav file. Track stats:
      - deleted          : xóa thành công
      - already_missing  : file đã không tồn tại (chạy lần 2 / đã xóa thủ công)
      - permission_error : Windows: file đang bị process khác giữ
      - other_errors     : các lỗi khác (path quá dài, disk error, ...)
      - bytes_freed      : tổng bytes đã giải phóng

    Return: dict stats + list các file fail (để ghi vào failed.txt).
    """
    try:
        from tqdm import tqdm
        iterator = tqdm(wavs, desc="Deleting", ncols=100, unit="file")
    except ImportError:
        logger.info("  (tqdm không cài — không có progress bar)")
        iterator = wavs

    stats = {
        "deleted": 0,
        "already_missing": 0,
        "permission_error": 0,
        "other_errors": 0,
        "bytes_freed": 0,
    }
    failed_files: List[Tuple[str, str]] = []  # (path, error_msg)

    if dry_run:
        prefix = "[DRY-RUN] "
    else:
        prefix = ""

    for wav_path in iterator:
        path_obj = Path(wav_path)

        if not path_obj.exists():
            stats["already_missing"] += 1
            continue

        if dry_run:
            # Chỉ đo size, không xóa
            try:
                stats["bytes_freed"] += path_obj.stat().st_size
                stats["deleted"] += 1
            except OSError:
                stats["other_errors"] += 1
            continue

        # Xóa thật
        try:
            size = path_obj.stat().st_size
            path_obj.unlink()
            stats["deleted"] += 1
            stats["bytes_freed"] += size
        except PermissionError as e:
            stats["permission_error"] += 1
            failed_files.append((wav_path, f"PermissionError: {e}"))
            if stats["permission_error"] <= 3:
                logger.warning(f"  Permission denied: {wav_path}")
        except OSError as e:
            stats["other_errors"] += 1
            failed_files.append((wav_path, f"OSError: {e}"))
            if stats["other_errors"] <= 3:
                logger.warning(f"  OSError xóa {wav_path}: {e}")
        except Exception as e:
            stats["other_errors"] += 1
            failed_files.append((wav_path, f"{type(e).__name__}: {e}"))
            if stats["other_errors"] <= 3:
                logger.warning(f"  Exception xóa {wav_path}: {e}")

    return stats, failed_files


def write_failed_report(
    failed_files: List[Tuple[str, str]],
    log_dir: Path,
    logger: logging.Logger,
):
    """Ghi danh sách file fail vào file riêng để retry sau."""
    if not failed_files:
        return

    failed_path = log_dir / "delete_unused_wavs_failed.txt"
    with open(failed_path, "w", encoding="utf-8") as f:
        f.write(f"# Files KHÔNG xóa được — generated by delete_unused_wavs.py\n")
        f.write(f"# Tổng: {len(failed_files):,}\n")
        f.write(f"# Format: <path>\\t<error>\n")
        f.write(f"#\n")
        f.write(f"# Để retry: dùng file path-only (col 1) làm input mới cho\n")
        f.write(f"# delete_unused_wavs.py (rename thành wavs_to_delete.txt)\n\n")
        for path, err in failed_files:
            f.write(f"{path}\t{err}\n")
    logger.info(f"  Failed list   : {failed_path}")


# =============================================================================
# MAIN FLOW
# =============================================================================

def run(config: DeleteConfig, logger: logging.Logger):
    """Quy trình chính."""
    output_dir = Path(config.output_dir)
    log_dir = Path(config.work_dir) / "logs"

    delete_path = output_dir / config.wavs_to_delete_file
    train_path = output_dir / config.train_list
    val_path = output_dir / config.val_list

    # =====================================================================
    # LỚP 1: Load + verify input file
    # =====================================================================
    logger.info("=" * 60)
    logger.info("  LỚP 1: Đọc wavs_to_delete.txt")
    logger.info("=" * 60)
    wavs_to_delete = load_wavs_to_delete(delete_path, logger)

    if not wavs_to_delete:
        logger.error("File wavs_to_delete.txt rỗng — không có gì để xóa.")
        return

    # Path sanity check (Windows vs Linux)
    if not check_path_compatibility(wavs_to_delete, logger):
        logger.error("ABORT: path mismatch (xem warning ở trên)")
        sys.exit(1)

    # =====================================================================
    # LỚP 2: Cross-check với filelist mới
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  LỚP 2: Cross-check với vivoice_train_list + vivoice_val_list")
    logger.info("=" * 60)
    wavs_to_keep = load_wavs_in_filelists(train_path, val_path, config.delimiter, logger)

    safe_to_delete = cross_check_overlap(wavs_to_delete, wavs_to_keep, logger)
    if not safe_to_delete:
        logger.error("ABORT: cross-check failed (xem lỗi ở trên)")
        sys.exit(2)

    # =====================================================================
    # Estimate size
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    logger.info("  Ước lượng dung lượng sẽ giải phóng")
    logger.info("=" * 60)
    estimated_bytes = estimate_total_size(safe_to_delete, logger)

    # =====================================================================
    # LỚP 3: Confirm prompt (skip nếu --yes hoặc --dry-run)
    # =====================================================================
    if not config.dry_run and not config.skip_confirm:
        if not confirm_deletion(len(safe_to_delete), estimated_bytes, output_dir, logger):
            logger.info("")
            logger.info("Hủy bỏ. Không có file nào bị xóa.")
            return

    # =====================================================================
    # LỚP 4: Dry-run hoặc xóa thật
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    if config.dry_run:
        logger.info("  [DRY-RUN] Đo dung lượng (KHÔNG xóa file)")
    else:
        logger.info(f"  XÓA {len(safe_to_delete):,} FILES")
    logger.info("=" * 60)

    start_time = time.time()
    stats, failed_files = delete_files(safe_to_delete, config.dry_run, logger)
    elapsed = time.time() - start_time

    # =====================================================================
    # Report
    # =====================================================================
    logger.info("")
    logger.info("=" * 60)
    if config.dry_run:
        logger.info("  KẾT QUẢ [DRY-RUN]")
    else:
        logger.info("  KẾT QUẢ")
    logger.info("=" * 60)
    logger.info(f"  Thời gian          : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Tổng input         : {len(safe_to_delete):,} files")
    if config.dry_run:
        logger.info(f"  Sẽ xóa             : {stats['deleted']:,} files")
        logger.info(f"  Sẽ giải phóng      : {format_size(stats['bytes_freed'])}")
    else:
        logger.info(f"  Đã xóa             : {stats['deleted']:,} files")
        logger.info(f"  Đã giải phóng      : {format_size(stats['bytes_freed'])}")
    logger.info(f"  Đã không tồn tại   : {stats['already_missing']:,} files")
    if stats['permission_error']:
        logger.info(f"  Permission errors  : {stats['permission_error']:,} files")
    if stats['other_errors']:
        logger.info(f"  Other errors       : {stats['other_errors']:,} files")

    if failed_files and not config.dry_run:
        write_failed_report(failed_files, log_dir, logger)


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Xóa các .wav không cần thiết theo wavs_to_delete.txt (4 lớp safety)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ log + đo dung lượng, KHÔNG xóa file thật (an toàn để test trước)",
    )
    parser.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip prompt 'DELETE' (cho tự động hóa). KHÔNG khuyến nghị lần đầu chạy.",
    )
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = DeleteConfig.from_yaml(str(config_path))

    if args.dry_run:
        config.dry_run = True
    if args.yes:
        config.skip_confirm = True

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  DELETE UNUSED WAVS")
    logger.info("=" * 60)
    logger.info(f"Config               : {config_path.resolve()}")
    logger.info(f"Output dir           : {config.output_dir}")
    logger.info(f"Wavs-to-delete file  : {config.wavs_to_delete_file}")
    logger.info(f"Cross-check train/val: {config.train_list}, {config.val_list}")
    logger.info(f"Dry-run              : {config.dry_run}")
    logger.info(f"Skip confirm         : {config.skip_confirm}")
    logger.info("")

    try:
        run(config, logger)
        logger.info("")
        logger.info("=" * 60)
        if config.dry_run:
            logger.info("  HOÀN TẤT [DRY-RUN]")
            logger.info("=" * 60)
            logger.info("")
            logger.info("Nếu số liệu OK, chạy thật bằng:")
            logger.info("  python delete_unused_wavs.py")
        else:
            logger.info("  HOÀN TẤT")
            logger.info("=" * 60)
    except KeyboardInterrupt:
        logger.warning("\nĐã dừng (Ctrl+C). Các file đã xóa KHÔNG thể khôi phục.")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()