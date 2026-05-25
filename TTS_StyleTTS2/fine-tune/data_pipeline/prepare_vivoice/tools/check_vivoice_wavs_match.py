# -*- coding: utf-8 -*-
"""
check_vivoice_wavs_match.py

Kiểm tra xem:
  (A) Các wav được tham chiếu trong vivoice_train_list.txt + vivoice_val_list.txt
có khớp hoàn toàn với:
  (B) Các file .wav thực tế trong folder vivoice_clean_wavs hay không.

Script này CHỈ KIỂM TRA, KHÔNG XÓA FILE.

Mặc định xử lý tốt case filelist có path dạng:
  output\\vivoice_clean_wavs\\vivoice_0702150.wav|phoneme|68

Cách chạy nhanh:
  python check_vivoice_wavs_match.py

Cách chạy tự chỉ định path:
  python check_vivoice_wavs_match.py ^
    --train-list "D:\\...\\output\\vivoice_train_list.txt" ^
    --val-list   "D:\\...\\output\\vivoice_val_list.txt" ^
    --wav-dir    "D:\\...\\output\\vivoice_clean_wavs"
"""

import os
import sys
import argparse
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple


# =============================================================================
# Windows UTF-8 fix
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
# DEFAULT PATHS — sửa ở đây nếu cần
# =============================================================================

DEFAULT_BASE_DIR = Path(
    r"D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2"
    r"\fine-tune\data_pipeline\prepare_vivoice"
)

DEFAULT_TRAIN_LIST = DEFAULT_BASE_DIR / "output" / "vivoice_train_list.txt"
DEFAULT_VAL_LIST = DEFAULT_BASE_DIR / "output" / "vivoice_val_list.txt"
DEFAULT_WAV_DIR = DEFAULT_BASE_DIR / "output" / "vivoice_clean_wavs"
DEFAULT_REPORT_DIR = DEFAULT_BASE_DIR / "workdir" / "logs"


# =============================================================================
# Helpers
# =============================================================================

def norm_slash(s: str) -> str:
    """Chuẩn hóa slash để so sánh path ổn định."""
    return s.strip().replace("\\", "/")


def normalize_key_from_any_path(path_str: str, wav_dir_name: str) -> str:
    """
    Convert path bất kỳ thành key so sánh ổn định.

    Mục tiêu:
      output\\vivoice_clean_wavs\\vivoice_0702150.wav
      D:\\...\\output\\vivoice_clean_wavs\\vivoice_0702150.wav
      vivoice_clean_wavs\\vivoice_0702150.wav
      vivoice_0702150.wav

    đều quy về:
      vivoice_clean_wavs/vivoice_0702150.wav

    Nếu có subfolder bên trong vivoice_clean_wavs thì vẫn giữ subpath.
    """
    p = norm_slash(path_str)
    p_lower = p.lower()
    marker = wav_dir_name.strip("/\\").lower()

    # Case 1: path có chứa ".../vivoice_clean_wavs/xxx.wav"
    token = "/" + marker + "/"
    idx = p_lower.rfind(token)
    if idx >= 0:
        return p_lower[idx + 1:]

    # Case 2: path bắt đầu bằng "vivoice_clean_wavs/xxx.wav"
    prefix = marker + "/"
    if p_lower.startswith(prefix):
        return p_lower

    # Case 3: path chỉ là filename hoặc dạng không chứa marker
    # Fallback: lấy basename và gắn vào wav_dir_name.
    filename = Path(p).name.lower()
    return f"{marker}/{filename}"


def disk_key_from_path(wav_path: Path, wav_dir: Path) -> str:
    """
    Convert file wav thực tế trên disk thành key:
      vivoice_clean_wavs/relative/path.wav
    """
    rel = wav_path.relative_to(wav_dir)
    key = Path(wav_dir.name) / rel
    return norm_slash(str(key)).lower()


def parse_filelist(
    filelist_path: Path,
    wav_dir_name: str,
    delimiter: str = "|",
) -> Tuple[List[Tuple[str, str, int]], List[Tuple[int, str, str]]]:
    """
    Đọc filelist, return:
      refs: list of (key, original_wav_path, line_no)
      bad_lines: list of (line_no, reason, content)
    """
    refs: List[Tuple[str, str, int]] = []
    bad_lines: List[Tuple[int, str, str]] = []

    if not filelist_path.exists():
        raise FileNotFoundError(f"Không tìm thấy filelist: {filelist_path}")

    with open(filelist_path, "r", encoding="utf-8-sig") as f:
        for line_no, raw in enumerate(f, start=1):
            line = raw.strip()

            if not line:
                continue
            if line.startswith("#"):
                continue

            if delimiter not in line:
                bad_lines.append((line_no, "Không có delimiter '|'", line[:200]))
                continue

            wav_part = line.split(delimiter, 1)[0].strip()

            if not wav_part:
                bad_lines.append((line_no, "Cột wav_path rỗng", line[:200]))
                continue

            if not wav_part.lower().endswith(".wav"):
                bad_lines.append((line_no, "wav_path không kết thúc bằng .wav", wav_part))
                continue

            key = normalize_key_from_any_path(wav_part, wav_dir_name)
            refs.append((key, wav_part, line_no))

    return refs, bad_lines


def scan_disk_wavs(wav_dir: Path) -> List[Tuple[str, Path]]:
    """
    Scan toàn bộ .wav trong wav_dir, recursive.
    Return list of (key, absolute_path).
    """
    if not wav_dir.exists():
        raise FileNotFoundError(f"Không tìm thấy wav_dir: {wav_dir}")
    if not wav_dir.is_dir():
        raise NotADirectoryError(f"wav_dir không phải folder: {wav_dir}")

    wavs: List[Tuple[str, Path]] = []
    for p in wav_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".wav":
            key = disk_key_from_path(p, wav_dir)
            wavs.append((key, p.resolve()))

    return wavs


def write_lines(path: Path, lines: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(str(line) + "\n")


def build_lookup(items: List[Tuple[str, object]]) -> Dict[str, List[object]]:
    d: Dict[str, List[object]] = defaultdict(list)
    for key, value in items:
        d[key].append(value)
    return d


# =============================================================================
# Main check
# =============================================================================

def run(args: argparse.Namespace) -> int:
    train_list = Path(args.train_list)
    val_list = Path(args.val_list)
    wav_dir = Path(args.wav_dir)
    report_dir = Path(args.report_dir)
    delimiter = args.delimiter

    print("=" * 80)
    print("CHECK VIVOICE WAVS MATCH")
    print("=" * 80)
    print(f"Train list : {train_list}")
    print(f"Val list   : {val_list}")
    print(f"Wav dir    : {wav_dir}")
    print(f"Report dir : {report_dir}")
    print("")

    wav_dir_name = wav_dir.name

    # -------------------------------------------------------------------------
    # 1. Load train/val references
    # -------------------------------------------------------------------------
    print("[1/4] Đọc train/val filelists...")

    train_refs, train_bad = parse_filelist(train_list, wav_dir_name, delimiter)
    val_refs, val_bad = parse_filelist(val_list, wav_dir_name, delimiter)

    all_refs = train_refs + val_refs
    all_bad = [("train", *x) for x in train_bad] + [("val", *x) for x in val_bad]

    ref_keys = [x[0] for x in all_refs]
    ref_set = set(ref_keys)
    ref_counter = Counter(ref_keys)

    print(f"  Train refs        : {len(train_refs):,}")
    print(f"  Val refs          : {len(val_refs):,}")
    print(f"  Total refs        : {len(all_refs):,}")
    print(f"  Unique ref wavs   : {len(ref_set):,}")
    print(f"  Bad lines         : {len(all_bad):,}")

    duplicate_ref_keys = sorted([k for k, c in ref_counter.items() if c > 1])
    print(f"  Duplicate refs    : {len(duplicate_ref_keys):,}")
    print("")

    # -------------------------------------------------------------------------
    # 2. Scan disk wavs
    # -------------------------------------------------------------------------
    print("[2/4] Scan file .wav thực tế trên disk...")

    disk_wavs = scan_disk_wavs(wav_dir)
    disk_keys = [x[0] for x in disk_wavs]
    disk_set = set(disk_keys)
    disk_counter = Counter(disk_keys)

    print(f"  Disk wav files    : {len(disk_wavs):,}")
    print(f"  Unique disk wavs  : {len(disk_set):,}")

    duplicate_disk_keys = sorted([k for k, c in disk_counter.items() if c > 1])
    print(f"  Duplicate on disk : {len(duplicate_disk_keys):,}")
    print("")

    # -------------------------------------------------------------------------
    # 3. Compare
    # -------------------------------------------------------------------------
    print("[3/4] So sánh filelist vs disk...")

    missing_on_disk = sorted(ref_set - disk_set)
    extra_on_disk = sorted(disk_set - ref_set)

    print(f"  Missing on disk   : {len(missing_on_disk):,}")
    print(f"    = Có trong train/val nhưng KHÔNG thấy file .wav trên disk")
    print(f"  Extra on disk     : {len(extra_on_disk):,}")
    print(f"    = Có trên disk nhưng KHÔNG được dùng trong train/val")
    print("")

    # Build lookups for full path output
    disk_lookup = build_lookup(disk_wavs)
    ref_lookup: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for key, original_wav_path, line_no in all_refs:
        ref_lookup[key].append((original_wav_path, line_no))

    # -------------------------------------------------------------------------
    # 4. Write reports
    # -------------------------------------------------------------------------
    print("[4/4] Ghi report...")

    report_dir.mkdir(parents=True, exist_ok=True)

    # Missing: keys + original refs
    missing_lines: List[str] = []
    for key in missing_on_disk:
        missing_lines.append(f"# {key}")
        for original_wav_path, line_no in ref_lookup.get(key, []):
            missing_lines.append(f"{original_wav_path}\tline={line_no}")
        missing_lines.append("")

    # Extra: absolute paths, useful for later delete review
    extra_abs_lines: List[str] = []
    for key in extra_on_disk:
        for abs_path in disk_lookup.get(key, []):
            extra_abs_lines.append(str(abs_path))

    # Duplicate refs
    duplicate_ref_lines: List[str] = []
    for key in duplicate_ref_keys:
        duplicate_ref_lines.append(f"# {key}  count={ref_counter[key]}")
        for original_wav_path, line_no in ref_lookup.get(key, []):
            duplicate_ref_lines.append(f"{original_wav_path}\tline={line_no}")
        duplicate_ref_lines.append("")

    # Bad lines
    bad_line_lines: List[str] = []
    for src, line_no, reason, content in all_bad:
        bad_line_lines.append(f"{src}\tline={line_no}\t{reason}\t{content}")

    # Summary report
    summary_path = report_dir / "check_vivoice_wavs_match_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("CHECK VIVOICE WAVS MATCH SUMMARY\n")
        f.write("=" * 80 + "\n")
        f.write(f"Train list          : {train_list}\n")
        f.write(f"Val list            : {val_list}\n")
        f.write(f"Wav dir             : {wav_dir}\n")
        f.write("\n")
        f.write(f"Train refs          : {len(train_refs):,}\n")
        f.write(f"Val refs            : {len(val_refs):,}\n")
        f.write(f"Total refs          : {len(all_refs):,}\n")
        f.write(f"Unique ref wavs     : {len(ref_set):,}\n")
        f.write(f"Disk wav files      : {len(disk_wavs):,}\n")
        f.write(f"Unique disk wavs    : {len(disk_set):,}\n")
        f.write("\n")
        f.write(f"Missing on disk     : {len(missing_on_disk):,}\n")
        f.write(f"Extra on disk       : {len(extra_on_disk):,}\n")
        f.write(f"Duplicate refs      : {len(duplicate_ref_keys):,}\n")
        f.write(f"Duplicate on disk   : {len(duplicate_disk_keys):,}\n")
        f.write(f"Bad lines           : {len(all_bad):,}\n")
        f.write("\n")

        if not missing_on_disk and not extra_on_disk:
            f.write("MATCH_RESULT        : OK - filelist và folder wav khớp nhau 100% theo set wav.\n")
        else:
            f.write("MATCH_RESULT        : NOT OK - có lệch giữa filelist và folder wav.\n")

        if duplicate_ref_keys:
            f.write("WARNING             : Có wav bị reference nhiều hơn 1 lần trong train+val.\n")
        if all_bad:
            f.write("WARNING             : Có dòng lỗi format trong train/val.\n")

    missing_path = report_dir / "check_missing_on_disk.txt"
    extra_path = report_dir / "check_extra_on_disk.txt"
    duplicate_refs_path = report_dir / "check_duplicate_refs.txt"
    bad_lines_path = report_dir / "check_bad_lines.txt"

    write_lines(missing_path, missing_lines)
    write_lines(extra_path, extra_abs_lines)
    write_lines(duplicate_refs_path, duplicate_ref_lines)
    write_lines(bad_lines_path, bad_line_lines)

    print(f"  Summary           : {summary_path}")
    print(f"  Missing report    : {missing_path}")
    print(f"  Extra report      : {extra_path}")
    print(f"  Duplicate refs    : {duplicate_refs_path}")
    print(f"  Bad lines         : {bad_lines_path}")
    print("")

    # -------------------------------------------------------------------------
    # Final result
    # -------------------------------------------------------------------------
    print("=" * 80)
    if not missing_on_disk and not extra_on_disk:
        print("✅ OK: train/val và folder vivoice_clean_wavs khớp nhau 100% theo set wav.")
        result_code = 0
    else:
        print("❌ NOT OK: train/val và folder vivoice_clean_wavs CHƯA khớp nhau.")
        if missing_on_disk:
            print(f"   - {len(missing_on_disk):,} wav có trong train/val nhưng thiếu trên disk.")
        if extra_on_disk:
            print(f"   - {len(extra_on_disk):,} wav thừa trên disk, không có trong train/val.")
        result_code = 2

    if duplicate_ref_keys:
        print(f"⚠️  Cảnh báo: {len(duplicate_ref_keys):,} wav bị lặp reference trong train+val.")
    if all_bad:
        print(f"⚠️  Cảnh báo: {len(all_bad):,} dòng lỗi format trong train/val.")

    print("=" * 80)

    return result_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check train/val filelists có khớp 100% với wav files trên disk không."
    )

    parser.add_argument(
        "--train-list",
        type=str,
        default=str(DEFAULT_TRAIN_LIST),
        help="Path tới output/vivoice_train_list.txt",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        default=str(DEFAULT_VAL_LIST),
        help="Path tới output/vivoice_val_list.txt",
    )
    parser.add_argument(
        "--wav-dir",
        type=str,
        default=str(DEFAULT_WAV_DIR),
        help="Path tới folder output/vivoice_clean_wavs",
    )
    parser.add_argument(
        "--report-dir",
        type=str,
        default=str(DEFAULT_REPORT_DIR),
        help="Folder để ghi các file report",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="Delimiter trong filelist, mặc định là '|'",
    )

    args = parser.parse_args()
    exit_code = run(args)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
