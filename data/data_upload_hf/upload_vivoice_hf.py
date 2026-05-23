from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

DEFAULT_LOCAL_DIR = r"D:/Documents/HUST/HUST_Project/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/output/vivoice_clean_wavs"
DEFAULT_REPO_ID = "vieanh/vivoice_clean_wavs"
DEFAULT_REPO_TYPE = "dataset"

def run_cmd(cmd: list[str], env: dict | None = None) -> None:
    print("\n[RUN]", " ".join(f'"{x}"' if " " in x else x for x in cmd))
    subprocess.run(cmd, check=True, env=env)


def human_size(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if value < 1024:
            return f"{value:.2f} {unit}"
        value /= 1024
    return f"{value:.2f} PB"


def ensure_hf_cli_available() -> None:
    if shutil.which("hf") is None:
        print(
            "\n[ERROR] Không tìm thấy lệnh `hf`.\n"
            "Hãy cài/cập nhật Hugging Face Hub trước:\n\n"
            '    py -m pip install -U huggingface_hub hf_xet\n'
        )
        sys.exit(1)

def ensure_hf_login() -> None:
    """
    Ưu tiên:
    1. Nếu đã set biến môi trường HF_TOKEN thì login bằng token đó.
    2. Nếu đã login sẵn thì dùng luôn.
    3. Nếu chưa login thì mở `hf auth login` để bạn paste token.
    """
    ensure_hf_cli_available()

    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print("[AUTH] Đã thấy biến môi trường HF_TOKEN. Đang chạy: hf auth login --token $HF_TOKEN")
        run_cmd(["hf", "auth", "login", "--token", hf_token])
        return

    check = subprocess.run(
        ["hf", "auth", "whoami"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    if check.returncode == 0:
        print("[AUTH] Đã login Hugging Face:")
        print(check.stdout.strip())
        return

    print(
        "\n[AUTH] Bạn chưa login Hugging Face.\n"
        "Script sẽ chạy `hf auth login`.\n"
        "Hãy paste Hugging Face User Access Token có quyền WRITE.\n"
    )
    run_cmd(["hf", "auth", "login"])

def scan_folder(folder: Path) -> dict:
    total_files = 0
    total_bytes = 0
    wav_files = 0
    wav_bytes = 0
    files_per_dir: Counter[str] = Counter()

    ignored_top_dirs = {".git", ".cache", "__pycache__"}

    for p in folder.rglob("*"):
        if not p.is_file():
            continue

        rel = p.relative_to(folder)
        if rel.parts and rel.parts[0] in ignored_top_dirs:
            continue

        try:
            size = p.stat().st_size
        except OSError:
            print(f"[WARN] Không đọc được file, bỏ qua khi scan: {p}")
            continue

        total_files += 1
        total_bytes += size

        parent_rel = str(p.parent.relative_to(folder))
        if parent_rel == ".":
            parent_rel = "<repo-root>"
        files_per_dir[parent_rel] += 1

        if p.suffix.lower() == ".wav":
            wav_files += 1
            wav_bytes += size

    largest_dir, largest_dir_count = ("", 0)
    if files_per_dir:
        largest_dir, largest_dir_count = files_per_dir.most_common(1)[0]

    return {
        "total_files": total_files,
        "total_bytes": total_bytes,
        "wav_files": wav_files,
        "wav_bytes": wav_bytes,
        "largest_dir": largest_dir,
        "largest_dir_count": largest_dir_count,
    }

def print_scan_report(folder: Path, stats: dict, only_wav: bool) -> None:
    print("\n========== LOCAL DATA REPORT ==========")
    print(f"Local folder : {folder}")
    print(f"Total files  : {stats['total_files']:,}")
    print(f"Total size   : {human_size(stats['total_bytes'])}")
    print(f"WAV files    : {stats['wav_files']:,}")
    print(f"WAV size     : {human_size(stats['wav_bytes'])}")
    print(f"Largest dir  : {stats['largest_dir']} ({stats['largest_dir_count']:,} files)")
    print(f"Upload mode  : {'ONLY .wav files' if only_wav else 'ALL files except ignored patterns'}")
    print("=======================================\n")

    if stats["total_files"] == 0:
        print("[ERROR] Folder không có file nào để upload.")
        sys.exit(1)

    if only_wav and stats["wav_files"] == 0:
        print("[ERROR] Bạn đang bật only-wav nhưng folder không có file .wav nào.")
        sys.exit(1)

    if stats["total_files"] > 100_000:
        print("[WARN] Tổng số file > 100k. Hugging Face có thể cảnh báo/recommend chia nhỏ cấu trúc repo.")

    if stats["largest_dir_count"] > 10_000:
        print(
            "[WARN] Có một thư mục chứa > 10k file. "
            "Với audio dataset lớn, nên chia thành subfolder/shard để repo dễ duyệt hơn."
        )

def upload_with_python_api(args, folder: Path) -> None:
    from huggingface_hub import HfApi

    api = HfApi()

    print("[HF] Tạo repo nếu chưa tồn tại...")
    create_kwargs = {
        "repo_id": args.repo_id,
        "repo_type": args.repo_type,
        "exist_ok": True,
    }
    if args.private:
        create_kwargs["private"] = True

    api.create_repo(**create_kwargs)

    allow_patterns = None
    if args.only_wav:
        allow_patterns = ["*.wav", "**/*.wav"]

    ignore_patterns = [
        ".git/**",
        ".cache/**",
        "__pycache__/**",
        "*.tmp",
        "*.part",
        "Thumbs.db",
        "desktop.ini",
    ]

    print("[HF] Bắt đầu upload bằng HfApi.upload_large_folder...")
    print("[HF] Nếu mất mạng hoặc Ctrl+C, chạy lại đúng lệnh này để resume.\n")

    api.upload_large_folder(
        repo_id=args.repo_id,
        repo_type=args.repo_type,
        folder_path=folder,
        revision=args.revision,
        private=True if args.private else None,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        num_workers=args.workers,
        print_report=True,
        print_report_every=args.report_every,
    )

    print("\n[DONE] Upload hoàn tất.")
    print(f"Repo: https://huggingface.co/datasets/{args.repo_id}")

def upload_with_hf_upload_large_folder(args, folder: Path) -> None:
    cmd = [
        "hf",
        "upload-large-folder",
        args.repo_id,
        str(folder),
        "--repo-type",
        args.repo_type,
        "--num-workers",
        str(args.workers),
    ]

    if args.revision:
        cmd += ["--revision", args.revision]

    print("[HF CLI] Bắt đầu upload bằng hf upload-large-folder...")
    run_cmd(cmd)
    print("\n[DONE] Upload hoàn tất.")

def upload_with_hf_upload(args, folder: Path) -> None:
    print(
        "\n[WARN] Bạn đang dùng `hf upload`. "
        "Với folder 135GB, mình không khuyến nghị cách này bằng upload_large_folder.\n"
    )

    cmd = [
        "hf",
        "upload",
        args.repo_id,
        str(folder),
        ".",
        "--repo-type",
        args.repo_type,
    ]

    if args.revision:
        cmd += ["--revision", args.revision]

    print("[HF CLI] Bắt đầu upload bằng hf upload...")
    run_cmd(cmd)
    print("\n[DONE] Upload hoàn tất.")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Push vivoice_clean_wavs lên Hugging Face Dataset repo."
    )

    parser.add_argument(
        "--local-dir",
        default=DEFAULT_LOCAL_DIR,
        help="Đường dẫn folder local chứa wav.",
    )
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face repo id, ví dụ: vieanh/vivoice_clean_wavs",
    )
    parser.add_argument(
        "--repo-type",
        default=DEFAULT_REPO_TYPE,
        choices=["dataset", "model", "space"],
        help="Loại repo trên Hugging Face.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Branch/revision muốn upload. Mặc định là main.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Số worker upload. Mạng yếu thì giảm còn 4; mạng khỏe có thể tăng 12/16.",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=60,
        help="Số giây giữa mỗi lần in tiến độ.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Tạo repo private nếu repo chưa tồn tại.",
    )
    parser.add_argument(
        "--only-wav",
        action="store_true",
        default=True,
        help="Chỉ upload file .wav. Mặc định bật.",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        help="Upload mọi file trong folder, không chỉ .wav.",
    )
    parser.add_argument(
        "--method",
        default="api-large",
        choices=["api-large", "hf-upload-large-folder", "hf-upload"],
        help=(
            "api-large: dùng Python HfApi.upload_large_folder; "
            "hf-upload-large-folder: dùng CLI mới; "
            "hf-upload: dùng hf upload mới nhưng không khuyến nghị cho 135GB."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ scan folder và in thống kê, chưa upload.",
    )

    return parser.parse_args()

def main() -> None:
    args = parse_args()

    if args.include_all:
        args.only_wav = False

    # Xet backend giúp upload nhanh hơn; high performance sẽ dùng mạnh CPU/băng thông hơn.
    os.environ.setdefault("HF_XET_HIGH_PERFORMANCE", "1")

    folder = Path(args.local_dir)

    if not folder.exists():
        print(f"[ERROR] Không tồn tại folder:\n{folder}")
        sys.exit(1)

    if not folder.is_dir():
        print(f"[ERROR] Đường dẫn không phải folder:\n{folder}")
        sys.exit(1)

    stats = scan_folder(folder)
    print_scan_report(folder, stats, only_wav=args.only_wav)

    if args.dry_run:
        print("[DRY-RUN] Chỉ kiểm tra folder, chưa upload.")
        return

    ensure_hf_login()

    if args.method == "api-large":
        upload_with_python_api(args, folder)
    elif args.method == "hf-upload-large-folder":
        upload_with_hf_upload_large_folder(args, folder)
    elif args.method == "hf-upload":
        upload_with_hf_upload(args, folder)
    else:
        raise ValueError(f"Unknown method: {args.method}")

if __name__ == "__main__":
    main()