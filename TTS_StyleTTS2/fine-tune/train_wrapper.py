"""
=============================================================================
  TRAIN WRAPPER — Nhạc trưởng điều phối 3 Giai đoạn Huấn luyện
=============================================================================
Mục tiêu: Đóng vai trò "nhạc trưởng" quản lý toàn bộ quy trình huấn luyện
          StyleTTS2 tiếng Việt 3 giai đoạn:
            Stage 1: Acoustic & Alignment (train_first.py)
            Stage 2: Expressive Training  (train_second.py)
            Stage 3: Fine-tune Bác Ngạn   (train_finetune.py)

Chức năng chính:
  1. Đọc phoneme_vocab.json → inject n_token thực tế vào config YAML
  2. Auto-chain: Tìm checkpoint tốt nhất từ giai đoạn trước → điền vào
     pretrained_model của giai đoạn sau
  3. Gọi subprocess tới script gốc của StyleTTS2 (không viết lại training loop)
  4. Log chi tiết và kiểm tra lỗi

Chạy lệnh:
    python train_wrapper.py --stage 1
    python train_wrapper.py --stage 2
    python train_wrapper.py --stage 3
    python train_wrapper.py --stage 1 --dry-run   (chỉ kiểm tra, không chạy)

Yêu cầu: Phải chạy từ thư mục fine-tune/ (hoặc chỉ định --project-root)
=============================================================================
"""

import os
import sys
import re
import json
import copy
import time
import shutil
import logging
import argparse
import subprocess
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

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
# CONFIGURATION
# =============================================================================

# Mapping stage → script gốc và config template
STAGE_MAP = {
    1: {
        "script": "train_first.py",
        "config_template": "config/config_stage1.yaml",
        "description": "Stage 1: Acoustic & Alignment (Base Vietnamese)",
        "log_dir_key": "log_dir",
    },
    2: {
        "script": "train_second.py",
        "config_template": "config/config_stage2.yaml",
        "description": "Stage 2: Expressive Training (JAT + SLM + OOD)",
        "log_dir_key": "log_dir",
    },
    3: {
        "script": "train_finetune.py",
        "config_template": "config/config_stage3.yaml",
        "description": "Stage 3: Fine-tune Giọng Bác Ngạn",
        "log_dir_key": "log_dir",
    },
}


@dataclass
class WrapperConfig:
    """Cấu hình cho train_wrapper."""

    # Đường dẫn project
    styletts2_root: str = ""        # Thư mục chứa repo gốc StyleTTS2
    finetune_root: str = ""         # Thư mục fine-tune/ hiện tại

    # Vocab
    vocab_file: str = ""            # phoneme_vocab.json

    # Stage cần chạy
    stage: int = 1

    # Override pretrained_model (bỏ qua auto-chain)
    pretrained_model: Optional[str] = None

    # Override batch_size
    batch_size: Optional[int] = None

    # Dry run (chỉ kiểm tra, không chạy subprocess)
    dry_run: bool = False

    # Python executable
    python_exe: str = sys.executable

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "WrapperConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        wrapper = full_config.get("train_wrapper", {})
        paths = full_config.get("paths", {})

        return cls(
            styletts2_root=wrapper.get("styletts2_root",
                                       paths.get("styletts2_root", cls.styletts2_root)),
            finetune_root=wrapper.get("finetune_root",
                                      paths.get("finetune_root", cls.finetune_root)),
            vocab_file=wrapper.get("vocab_file",
                                   paths.get("vocab_file", cls.vocab_file)),
        )


# =============================================================================
# LOGGING SETUP
# =============================================================================

def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "train_wrapper.log"

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
    return logging.getLogger("train_wrapper")


# =============================================================================
# CORE: VOCAB INJECTION
# =============================================================================

def load_n_token(vocab_file: str, logger: logging.Logger) -> int:
    """
    Đọc phoneme_vocab.json và trả về n_token thực tế.
    Đây là giá trị sẽ thay thế placeholder 178 trong config.
    """
    vocab_path = Path(vocab_file)
    if not vocab_path.exists():
        logger.error(f"Không tìm thấy vocab file: {vocab_path}")
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab_data = json.load(f)

    n_token = vocab_data.get("n_token", None)
    if n_token is None:
        # Fallback: đếm từ char_to_id
        char_to_id = vocab_data.get("char_to_id", {})
        n_token = len(char_to_id)

    logger.info(f"Loaded n_token = {n_token} từ {vocab_path.name}")
    return n_token


def inject_n_token(config_dict: dict, n_token: int, logger: logging.Logger) -> dict:
    """
    Inject n_token thực tế vào config dict.
    Thay thế giá trị placeholder tại model_params.n_token.
    """
    config = copy.deepcopy(config_dict)

    old_value = config.get("model_params", {}).get("n_token", "N/A")

    if "model_params" in config:
        config["model_params"]["n_token"] = n_token
        logger.info(f"Injected n_token: {old_value} → {n_token}")
    else:
        logger.warning("Không tìm thấy key 'model_params' trong config!")

    return config


# =============================================================================
# CORE: AUTO-CHAIN CHECKPOINT
# =============================================================================

def find_best_checkpoint(log_dir: str, logger: logging.Logger) -> Optional[str]:
    """
    Quét thư mục log_dir của giai đoạn trước, tìm checkpoint tốt nhất.

    Heuristic ưu tiên:
    1. File có tên chứa "best" → chọn ngay
    2. File .pth mới nhất (theo modification time)

    Returns:
        Đường dẫn tuyệt đối tới checkpoint, hoặc None nếu không tìm thấy.
    """
    log_path = Path(log_dir)
    if not log_path.exists():
        logger.warning(f"Thư mục log không tồn tại: {log_path}")
        return None

    # Tìm tất cả file .pth
    pth_files = list(log_path.rglob("*.pth"))
    if not pth_files:
        logger.warning(f"Không tìm thấy file .pth trong: {log_path}")
        return None

    logger.info(f"Tìm thấy {len(pth_files)} checkpoint(s) trong {log_path}")

    # Ưu tiên 1: File có "best" trong tên
    best_files = [f for f in pth_files if "best" in f.name.lower()]
    if best_files:
        chosen = str(best_files[0].resolve())
        logger.info(f"  Chọn (best): {best_files[0].name}")
        return chosen

    # Ưu tiên 2: File mới nhất
    pth_files_sorted = sorted(pth_files, key=lambda f: f.stat().st_mtime, reverse=True)
    chosen = str(pth_files_sorted[0].resolve())
    logger.info(f"  Chọn (mới nhất): {pth_files_sorted[0].name}")

    return chosen


def auto_chain_checkpoint(
    config_dict: dict,
    stage: int,
    finetune_root: str,
    override_pretrained: Optional[str],
    logger: logging.Logger,
) -> dict:
    """
    Tự động tìm checkpoint từ giai đoạn trước và điền vào pretrained_model.
    """
    config = copy.deepcopy(config_dict)

    # Nếu user override → dùng ngay
    if override_pretrained:
        if not Path(override_pretrained).exists():
            logger.error(f"Override checkpoint không tồn tại: {override_pretrained}")
            raise FileNotFoundError(f"Checkpoint not found: {override_pretrained}")
        config["pretrained_model"] = override_pretrained
        logger.info(f"Dùng override checkpoint: {override_pretrained}")
        return config

    # Stage 1: Không cần pretrained (train từ đầu)
    if stage == 1:
        config["pretrained_model"] = ""
        logger.info("Stage 1: Train từ đầu (không cần pretrained)")
        return config

    # Stage 2: Tìm checkpoint từ Stage 1
    if stage == 2:
        prev_config_path = Path(finetune_root) / "config" / "config_stage1.yaml"
        if prev_config_path.exists():
            with open(prev_config_path, "r", encoding="utf-8") as f:
                prev_config = yaml.safe_load(f)
            prev_log_dir = prev_config.get("log_dir", "")
            # Resolve relative to styletts2_root
            if prev_log_dir and not Path(prev_log_dir).is_absolute():
                prev_log_dir = str(Path(finetune_root).parent / "StyleTTS2" / prev_log_dir)
        else:
            prev_log_dir = ""

    # Stage 3: Tìm checkpoint từ Stage 2
    elif stage == 3:
        prev_config_path = Path(finetune_root) / "config" / "config_stage2.yaml"
        if prev_config_path.exists():
            with open(prev_config_path, "r", encoding="utf-8") as f:
                prev_config = yaml.safe_load(f)
            prev_log_dir = prev_config.get("log_dir", "")
            if prev_log_dir and not Path(prev_log_dir).is_absolute():
                prev_log_dir = str(Path(finetune_root).parent / "StyleTTS2" / prev_log_dir)
        else:
            prev_log_dir = ""
    else:
        prev_log_dir = ""

    # Tìm checkpoint
    if prev_log_dir:
        ckpt_path = find_best_checkpoint(prev_log_dir, logger)
        if ckpt_path:
            config["pretrained_model"] = ckpt_path
            logger.info(f"Auto-chain: pretrained_model = {ckpt_path}")
        else:
            logger.warning("Không tìm thấy checkpoint từ giai đoạn trước!")
            logger.warning("Hãy chỉ định thủ công bằng --pretrained-model")
    else:
        logger.warning(f"Không xác định được log_dir của giai đoạn trước (stage {stage - 1})")

    return config


# =============================================================================
# CORE: PREPARE & RUN
# =============================================================================

def prepare_config(
    stage: int,
    wrapper_config: WrapperConfig,
    logger: logging.Logger,
) -> Path:
    """
    Chuẩn bị config YAML cho stage:
    1. Đọc config template
    2. Inject n_token
    3. Auto-chain checkpoint
    4. Override batch_size (nếu có)
    5. Lưu config đã xử lý ra file tạm

    Returns:
        Path tới file config đã xử lý (sẵn sàng truyền cho script gốc)
    """
    stage_info = STAGE_MAP[stage]
    template_path = Path(wrapper_config.finetune_root) / stage_info["config_template"]

    if not template_path.exists():
        raise FileNotFoundError(f"Config template không tồn tại: {template_path}")

    # Đọc template
    with open(template_path, "r", encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)

    logger.info(f"Loaded config template: {template_path}")

    # 1. Inject n_token
    if wrapper_config.vocab_file:
        n_token = load_n_token(wrapper_config.vocab_file, logger)
        config_dict = inject_n_token(config_dict, n_token, logger)

    # 2. Auto-chain checkpoint
    config_dict = auto_chain_checkpoint(
        config_dict,
        stage,
        wrapper_config.finetune_root,
        wrapper_config.pretrained_model,
        logger,
    )

    # 3. Override batch_size
    if wrapper_config.batch_size is not None:
        old_bs = config_dict.get("batch_size", "N/A")
        config_dict["batch_size"] = wrapper_config.batch_size
        logger.info(f"Override batch_size: {old_bs} → {wrapper_config.batch_size}")

    # 4. Resolve relative paths (data_params) thành absolute paths
    config_dict = resolve_data_paths(config_dict, wrapper_config.finetune_root, logger)

    # 5. Lưu config đã xử lý
    processed_dir = Path(wrapper_config.finetune_root) / "config" / "_processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    processed_path = processed_dir / f"config_stage{stage}_processed.yaml"

    with open(processed_path, "w", encoding="utf-8") as f:
        yaml.dump(config_dict, f, default_flow_style=False, allow_unicode=True)

    logger.info(f"Saved processed config: {processed_path}")
    return processed_path


def resolve_data_paths(config_dict: dict, finetune_root: str, logger: logging.Logger) -> dict:
    """
    Chuyển các đường dẫn tương đối trong data_params thành absolute paths.
    Đường dẫn tương đối được resolve dựa trên finetune_root.
    """
    config = copy.deepcopy(config_dict)
    root = Path(finetune_root)

    data_params = config.get("data_params", {})
    path_keys = ["train_data", "val_data", "OOD_data"]

    for key in path_keys:
        value = data_params.get(key, "")
        if value and not Path(value).is_absolute():
            resolved = str((root / value).resolve())
            data_params[key] = resolved
            logger.info(f"  Resolved {key}: {value} → {resolved}")

    # Resolve PLBERT_dir
    plbert_dir = config.get("PLBERT_dir", "")
    if plbert_dir and not Path(plbert_dir).is_absolute():
        resolved = str((root / plbert_dir).resolve())
        config["PLBERT_dir"] = resolved
        logger.info(f"  Resolved PLBERT_dir: {plbert_dir} → {resolved}")

    # Resolve log_dir (nằm trong StyleTTS2 root)
    log_dir = config.get("log_dir", "")
    if log_dir and not Path(log_dir).is_absolute():
        styletts2_root = root.parent / "StyleTTS2"
        resolved = str((styletts2_root / log_dir).resolve())
        config["log_dir"] = resolved
        logger.info(f"  Resolved log_dir: {log_dir} → {resolved}")

    return config


def run_training(
    stage: int,
    processed_config_path: Path,
    wrapper_config: WrapperConfig,
    logger: logging.Logger,
):
    """
    Chạy script gốc của StyleTTS2 qua subprocess.
    Working directory = thư mục repo gốc StyleTTS2.
    """
    stage_info = STAGE_MAP[stage]
    script_name = stage_info["script"]

    styletts2_root = Path(wrapper_config.styletts2_root)
    script_path = styletts2_root / script_name

    if not script_path.exists():
        raise FileNotFoundError(
            f"Script gốc không tồn tại: {script_path}\n"
            f"Kiểm tra lại đường dẫn styletts2_root: {styletts2_root}"
        )

    # Xây dựng lệnh
    cmd = [
        wrapper_config.python_exe,
        str(script_path),
        "--config_path", str(processed_config_path.resolve()),
    ]

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"  CHẠY: {stage_info['description']}")
    logger.info("=" * 60)
    logger.info(f"  Script  : {script_path}")
    logger.info(f"  Config  : {processed_config_path}")
    logger.info(f"  CWD     : {styletts2_root}")
    logger.info(f"  Command : {' '.join(cmd)}")
    logger.info("=" * 60)

    if wrapper_config.dry_run:
        logger.info("  [DRY RUN] Không thực sự chạy subprocess.")
        logger.info("  Kiểm tra config đã xử lý tại:")
        logger.info(f"    {processed_config_path}")
        return

    # Chạy subprocess
    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            cwd=str(styletts2_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        # Stream output realtime
        for line in process.stdout:
            line = line.rstrip()
            if line:
                print(line)
                # Ghi vào log file (không dùng logger để tránh duplicate format)
                for handler in logger.handlers:
                    if isinstance(handler, logging.FileHandler):
                        handler.stream.write(line + "\n")
                        handler.stream.flush()

        process.wait()
        elapsed = time.time() - start_time

        if process.returncode == 0:
            logger.info("")
            logger.info("=" * 60)
            logger.info(f"  {stage_info['description']} — HOÀN TẤT!")
            logger.info(f"  Thời gian: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")
            logger.info("=" * 60)
        else:
            logger.error("")
            logger.error("=" * 60)
            logger.error(f"  {stage_info['description']} — THẤT BẠI!")
            logger.error(f"  Return code: {process.returncode}")
            logger.error(f"  Thời gian: {elapsed:.0f}s")
            logger.error("=" * 60)
            sys.exit(process.returncode)

    except KeyboardInterrupt:
        logger.warning("\nNhận tín hiệu Ctrl+C — Dừng training...")
        process.terminate()
        process.wait(timeout=10)
        logger.warning("Training đã dừng.")
        sys.exit(1)


# =============================================================================
# PRE-FLIGHT CHECKS
# =============================================================================

def preflight_checks(stage: int, wrapper_config: WrapperConfig, logger: logging.Logger):
    """Kiểm tra tất cả điều kiện trước khi chạy."""
    errors = []

    # 1. Kiểm tra styletts2_root
    root = Path(wrapper_config.styletts2_root)
    if not root.exists():
        errors.append(f"styletts2_root không tồn tại: {root}")
    else:
        script_name = STAGE_MAP[stage]["script"]
        if not (root / script_name).exists():
            errors.append(f"Script gốc không tồn tại: {root / script_name}")

    # 2. Kiểm tra config template
    ft_root = Path(wrapper_config.finetune_root)
    template = ft_root / STAGE_MAP[stage]["config_template"]
    if not template.exists():
        errors.append(f"Config template không tồn tại: {template}")

    # 3. Kiểm tra vocab file
    if wrapper_config.vocab_file:
        if not Path(wrapper_config.vocab_file).exists():
            errors.append(f"Vocab file không tồn tại: {wrapper_config.vocab_file}")

    # 4. Kiểm tra CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            logger.warning("CUDA không khả dụng! Training sẽ chạy trên CPU (rất chậm).")
        else:
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_mem / 1e9
            logger.info(f"GPU: {gpu_name} ({vram:.1f} GB VRAM)")
    except ImportError:
        errors.append("PyTorch chưa được cài đặt!")

    # 5. Kiểm tra data files
    if template.exists():
        with open(template, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        data_params = config.get("data_params", {})
        for key in ["train_data", "val_data"]:
            data_path = data_params.get(key, "")
            if data_path:
                resolved = Path(data_path)
                if not resolved.is_absolute():
                    resolved = ft_root / data_path
                if not resolved.exists():
                    logger.warning(f"Data file chưa tồn tại: {resolved} (key: {key})")

    if errors:
        logger.error("")
        logger.error("PRE-FLIGHT CHECK THẤT BẠI:")
        for err in errors:
            logger.error(f"  - {err}")
        logger.error("")
        sys.exit(1)
    else:
        logger.info("Pre-flight checks: PASSED")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Train Wrapper — Nhạc trưởng điều phối 3 Giai đoạn Huấn luyện StyleTTS2"
    )
    parser.add_argument(
        "--stage", "-s",
        type=int,
        required=True,
        choices=[1, 2, 3],
        help="Giai đoạn cần chạy: 1 (Acoustic), 2 (Expressive), 3 (Fine-tune Ngạn)",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Đường dẫn tới wrapper config YAML (tùy chọn)",
    )
    parser.add_argument(
        "--styletts2-root",
        type=str,
        default=None,
        help="Override đường dẫn repo gốc StyleTTS2",
    )
    parser.add_argument(
        "--vocab-file",
        type=str,
        default=None,
        help="Override đường dẫn phoneme_vocab.json",
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="Override checkpoint pretrained (bỏ qua auto-chain)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ chuẩn bị config, không chạy training",
    )
    args = parser.parse_args()

    # --- Load .env ---
    env_candidates = [Path(".env"), Path("../.env")]
    for env_path in env_candidates:
        if env_path.exists():
            load_dotenv(str(env_path))
            break

    # --- Load config ---
    if args.config and Path(args.config).exists():
        wrapper_config = WrapperConfig.from_yaml(args.config)
    else:
        wrapper_config = WrapperConfig()

    # --- Infer paths nếu chưa có ---
    # finetune_root = thư mục chứa train_wrapper.py
    if not wrapper_config.finetune_root:
        wrapper_config.finetune_root = str(Path(__file__).parent.resolve())

    # styletts2_root = thư mục anh em bên cạnh fine-tune/
    if not wrapper_config.styletts2_root:
        wrapper_config.styletts2_root = str(
            (Path(wrapper_config.finetune_root).parent / "StyleTTS2").resolve()
        )

    # vocab_file = tìm trong data_pipeline output
    if not wrapper_config.vocab_file:
        candidates = [
            Path(wrapper_config.finetune_root) / "data_pipeline" / "prepare_vicoice" / "output" / "phoneme_vocab.json",
            Path(wrapper_config.finetune_root) / "output" / "phoneme_vocab.json",
        ]
        for c in candidates:
            if c.exists():
                wrapper_config.vocab_file = str(c)
                break

    # --- Override từ CLI ---
    wrapper_config.stage = args.stage
    wrapper_config.dry_run = args.dry_run

    if args.styletts2_root:
        wrapper_config.styletts2_root = args.styletts2_root
    if args.vocab_file:
        wrapper_config.vocab_file = args.vocab_file
    if args.pretrained_model:
        wrapper_config.pretrained_model = args.pretrained_model
    if args.batch_size:
        wrapper_config.batch_size = args.batch_size

    # --- Setup logging ---
    log_dir = Path(wrapper_config.finetune_root) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    stage_info = STAGE_MAP[args.stage]
    logger.info("")
    logger.info("=" * 60)
    logger.info("  TRAIN WRAPPER — NHẠC TRƯỞNG STYLETTS2")
    logger.info("=" * 60)
    logger.info(f"  Stage           : {args.stage} — {stage_info['description']}")
    logger.info(f"  StyleTTS2 root  : {wrapper_config.styletts2_root}")
    logger.info(f"  Fine-tune root  : {wrapper_config.finetune_root}")
    logger.info(f"  Vocab file      : {wrapper_config.vocab_file or '(không chỉ định)'}")
    logger.info(f"  Pretrained      : {wrapper_config.pretrained_model or '(auto-chain)'}")
    logger.info(f"  Batch size      : {wrapper_config.batch_size or '(theo config)'}")
    logger.info(f"  Dry run         : {wrapper_config.dry_run}")
    logger.info(f"  Python          : {wrapper_config.python_exe}")
    logger.info("=" * 60)

    # --- Pre-flight checks ---
    logger.info("")
    logger.info("Kiểm tra điều kiện...")
    preflight_checks(args.stage, wrapper_config, logger)

    # --- Chuẩn bị config ---
    logger.info("")
    logger.info("Chuẩn bị config...")
    processed_config = prepare_config(args.stage, wrapper_config, logger)

    # --- Chạy training ---
    run_training(args.stage, processed_config, wrapper_config, logger)

    # --- Gợi ý bước tiếp theo ---
    logger.info("")
    if args.stage < 3 and not wrapper_config.dry_run:
        next_stage = args.stage + 1
        next_info = STAGE_MAP[next_stage]
        logger.info(f"Bước tiếp theo: python train_wrapper.py --stage {next_stage}")
        logger.info(f"  ({next_info['description']})")
    elif args.stage == 3 and not wrapper_config.dry_run:
        logger.info("TẤT CẢ 3 GIAI ĐOẠN ĐÃ HOÀN TẤT!")
        logger.info("Bước tiếp theo:")
        logger.info("  1. python create_mean_style.py   (trích xuất mean style vector)")
        logger.info("  2. python nlp_generator.py       (Phase 1: Qwen → script.json)")
        logger.info("  3. python tts_generator.py       (Phase 2: TTS → audiobook)")

if __name__ == "__main__":
    main()