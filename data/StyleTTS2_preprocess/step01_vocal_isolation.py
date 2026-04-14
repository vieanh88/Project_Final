"""
=============================================================
  BƯỚC 1: VOCAL ISOLATION — Demucs htdemucs_ft
=============================================================
Mục tiêu: Tách giọng nói ra khỏi nhạc nền và SFX trong audio gốc.

Input  : raw_audio/  (*.mp3, *.wav, *.flac, *.m4a)
Output : workdir/step01_vocals/  (vocals.wav cho mỗi file)

Cách chạy:
  python step01_vocal_isolation.py
  python step01_vocal_isolation.py --input raw_audio/ten_file.mp3   # 1 file cụ thể
  python step01_vocal_isolation.py --dry-run                        # xem danh sách file mà không chạy

Lưu ý:
  - Demucs htdemucs_ft cần ~2-3GB VRAM
  - Nếu lỗi CUDA OOM → đặt cpu_only: true trong config.yaml
  - Kết quả được cache: file đã tách rồi sẽ không tách lại (trừ khi xóa workdir)
=============================================================
"""
import os
import sys
import time
import logging
import argparse
import subprocess
import shutil
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm
from tqdm.auto import tqdm

import yaml

#  CONFIG LOADER
def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        print("      Hãy đảm bảo config.yaml ở cùng thư mục với script.")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

#  LOGGING
def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, encoding="utf-8", mode="a"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers,
    )
    return logging.getLogger("step01")

#  HELPERS
SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac", ".opus"}

def get_audio_files(input_dir: str, single_file: Optional[str] = None) -> List[Path]:
    """Lấy danh sách file audio cần xử lý."""
    if single_file:
        p = Path(single_file)
        if not p.exists():
            print(f"[LỖI] File không tồn tại: {single_file}")
            sys.exit(1)
        return [p]

    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"[LỖI] Thư mục input không tồn tại: {input_dir}")
        print(f"      Hãy tạo thư mục và đặt file audio vào đó.")
        sys.exit(1)

    files = sorted([
        f for f in input_path.iterdir()
        if f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])
    return files

def find_vocals_output(demucs_out_dir: Path, model: str, stem_name: str) -> Optional[Path]:
    """
    Demucs tạo cấu trúc: <out_dir>/<model>/<stem>/vocals.wav
    Tìm file vocals.wav dù cấu trúc thư mục có thay đổi nhỏ.
    """
    # Thử đường dẫn chuẩn trước
    standard = demucs_out_dir / model / stem_name / "vocals.wav"
    if standard.exists():
        return standard

    # Fallback: tìm bất kỳ vocals.wav nào trong demucs_out_dir
    matches = list(demucs_out_dir.rglob("vocals.wav"))
    # Lọc theo stem name để tránh nhầm lẫn khi xử lý nhiều file
    for m in matches:
        if stem_name.lower() in str(m).lower():
            return m

    # Nếu chỉ có 1 kết quả duy nhất thì trả về
    if len(matches) == 1:
        return matches[0]

    return None

def format_duration(seconds: float) -> str:
    """Định dạng thời gian đẹp hơn."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"

#  CORE: VOCAL SEPARATION
def separate_vocals(
    audio_path: Path,
    output_dir: Path,
    model: str,
    cpu_only: bool,
    logger: logging.Logger,
) -> Optional[Path]:
    """
    Chạy Demucs để tách vocals từ 1 file audio.

    Returns:
        Đường dẫn vocals.wav nếu thành công, None nếu thất bại.
    """
    stem = audio_path.stem
    demucs_out_dir = output_dir / "_demucs_raw"  # Demucs output thô
    demucs_out_dir.mkdir(parents=True, exist_ok=True)

    # Xác định device
    device = "cpu" if cpu_only else _get_device()

    logger.info(f"  Chạy Demucs [{model}] trên [{device}] ...")

    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"  # Ép Demucs (tiến trình con) sử dụng UTF-8

    cmd = [
        sys.executable, "-m", "demucs",
        "--two-stems=vocals",   # Chỉ tách vocals + accompaniment (nhanh hơn 4-stem)
        "-n", model,
        "-d", device,
        "--out", str(demucs_out_dir),
        str(audio_path),
    ]

    t_start = time.time()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    elapsed = time.time() - t_start

    if proc.returncode != 0:
        logger.error(f"  Demucs thất bại (returncode={proc.returncode}):")
        # In stderr để debug nhưng giới hạn độ dài
        err_lines = proc.stderr.strip().splitlines()
        for line in err_lines[-10:]:  # Chỉ in 10 dòng cuối
            logger.error(f"    {line}")
        return None

    # Tìm file vocals.wav output
    vocals_path = find_vocals_output(demucs_out_dir, model, stem)
    if vocals_path is None:
        logger.error(f"  Không tìm thấy vocals.wav sau khi Demucs chạy xong.")
        logger.error(f"  Kiểm tra thư mục: {demucs_out_dir}")
        return None

    logger.info(f"  Demucs xong ({format_duration(elapsed)}) → {vocals_path.name}")
    return vocals_path

def _get_device() -> str:
    """Kiểm tra GPU có available không."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Kiểm tra VRAM tổng
            total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return "cuda"
        return "cpu"
    except ImportError:
        return "cpu"

def copy_vocals_to_step_dir(
    vocals_path: Path,
    step_output_dir: Path,
    source_stem: str,
) -> Path:
    """
    Copy vocals.wav ra thư mục step01_vocals/ với tên rõ ràng.
    Demucs tạo cấu trúc lồng nhau → flatten ra cho dễ dùng ở bước sau.
    """
    dest = step_output_dir / f"{source_stem}_vocals.wav"
    shutil.copy2(str(vocals_path), str(dest))
    return dest

#  MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 1: Tách vocals bằng Demucs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", default="config.yaml",
        help="Đường dẫn file config (mặc định: config.yaml)"
    )
    parser.add_argument(
        "--input", default=None,
        help="Chỉ xử lý 1 file cụ thể (mặc định: xử lý tất cả trong input_dir)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Liệt kê file sẽ xử lý mà không thực sự chạy"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg = cfg["step01"]

    # Setup paths
    work_dir = Path(paths_cfg["work_dir"])
    step_output_dir = work_dir / "step01_vocals"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step01.log")

    logger = setup_logging(log_path)

    # Header
    logger.info("=" * 60)
    logger.info("  BƯỚC 1: VOCAL ISOLATION (Demucs)")
    logger.info("=" * 60)
    logger.info(f"  Config   : {args.config}")
    logger.info(f"  Input    : {paths_cfg['input_dir']}")
    logger.info(f"  Output   : {step_output_dir}")
    logger.info(f"  Model    : {step_cfg['demucs_model']}")
    logger.info(f"  CPU only : {step_cfg['cpu_only']}")

    # Kiểm tra Demucs đã cài chưa
    try:
        result = subprocess.run(
            [sys.executable, "-m", "demucs", "--help"],
            capture_output=True, timeout=10
        )
        if result.returncode != 0:
            raise RuntimeError()
    except Exception:
        logger.error("Demucs chưa được cài đặt!")
        logger.error("Cài đặt: pip install demucs")
        sys.exit(1)

    # Lấy danh sách file
    audio_files = get_audio_files(paths_cfg["input_dir"], args.input)

    if not audio_files:
        logger.warning(f"Không tìm thấy file audio trong: {paths_cfg['input_dir']}")
        logger.warning(f"Định dạng hỗ trợ: {', '.join(SUPPORTED_EXTENSIONS)}")
        sys.exit(0)

    logger.info(f"  Tìm thấy : {len(audio_files)} file audio")

    # Dry run
    if args.dry_run:
        logger.info("\n[DRY RUN] Danh sách file sẽ xử lý:")
        for i, f in enumerate(audio_files, 1):
            size_mb = f.stat().st_size / 1024**2
            out_path = step_output_dir / f"{f.stem}_vocals.wav"
            status = "✓ đã có" if out_path.exists() else "→ cần xử lý"
            logger.info(f"  {i:3d}. {f.name} ({size_mb:.1f} MB) — {status}")
        sys.exit(0)

    # Processing loop
    results = {"success": [], "skipped": [], "failed": []}
    total_start = time.time()

    # Tạo một thanh tiến trình tổng thể
    pbar = tqdm(audio_files, desc="Đang tách Vocal", unit="file", colour='green')

    for idx, audio_file in enumerate(audio_files, 1):
        stem = audio_file.stem

        # Cập nhật mô tả trên thanh tiến trình để biết đang làm file nào
        pbar.set_description(f"Đang xử lý: {audio_file.name[:20]}...")

        logger.info(f"\n[{idx}/{len(audio_files)}] {audio_file.name}")
        logger.info(f"  Size: {audio_file.stat().st_size / 1024**2:.1f} MB")

        # Kiểm tra cache
        final_output = step_output_dir / f"{stem}_vocals.wav"
        if step_cfg["skip_existing"] and final_output.exists():
            size_mb = final_output.stat().st_size / 1024**2
            logger.info(f"  Skip (đã có cache): {final_output.name} ({size_mb:.1f} MB)")
            results["skipped"].append(audio_file.name)
            continue

        file_start = time.time()

        # Chạy Demucs
        vocals_raw = separate_vocals(
            audio_path=audio_file,
            output_dir=step_output_dir,
            model=step_cfg["demucs_model"],
            cpu_only=step_cfg["cpu_only"],
            logger=logger,
        )

        if vocals_raw is None:
            logger.error(f"  ✗ THẤT BẠI: {audio_file.name}")
            results["failed"].append(audio_file.name)
            continue

        # Copy ra thư mục step01_vocals/ với tên đơn giản
        final_output = copy_vocals_to_step_dir(
            vocals_path=vocals_raw,
            step_output_dir=step_output_dir,
            source_stem=stem,
        )

        elapsed = time.time() - file_start
        size_mb = final_output.stat().st_size / 1024**2
        logger.info(
            f"  ✓ Xong ({format_duration(elapsed)}) → "
            f"{final_output.name} ({size_mb:.1f} MB)"
        )
        results["success"].append(audio_file.name)

    # Tổng kết
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 1")
    logger.info("=" * 60)
    logger.info(f"  ✓ Thành công : {len(results['success'])} file")
    logger.info(f"  ⏭ Bỏ qua    : {len(results['skipped'])} file (cache)")
    logger.info(f"  ✗ Thất bại  : {len(results['failed'])} file")
    logger.info(f"  Tổng thời gian: {format_duration(total_elapsed)}")

    if results["failed"]:
        logger.warning("\nCác file thất bại:")
        for f in results["failed"]:
            logger.warning(f"  - {f}")
        logger.warning("\nGợi ý xử lý lỗi:")
        logger.warning("  - CUDA OOM → đặt cpu_only: true trong config.yaml")
        logger.warning("  - Lỗi format → chuyển file sang .wav trước bằng ffmpeg")

    # Liệt kê output
    output_files = sorted(step_output_dir.glob("*_vocals.wav"))
    logger.info(f"\n  Output ({len(output_files)} file):")
    for f in output_files:
        logger.info(f"    {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

    if results["failed"]:
        sys.exit(1)

    logger.info(f"\n→ Chạy tiếp: python step02_audio_restoration.py")

if __name__ == "__main__":
    main()