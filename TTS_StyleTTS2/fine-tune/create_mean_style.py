"""
=============================================================================
  CREATE MEAN STYLE — Trích xuất Mean Style Vector giọng Bác Ngạn
=============================================================================
Mục tiêu: Chạy MỘT LẦN DUY NHẤT (offline) sau khi train xong Stage 3.
          Trích xuất Mean Style Vector đại diện cho giọng đọc mộc mạc
          chuẩn mực nhất của nghệ sĩ Nguyễn Ngọc Ngạn.

Logic:
  1. Load checkpoint StyleTTS2 (Stage 3)
  2. Đọc filelist ngan_train_phoneme.txt
  3. Chọn ngẫu nhiên ~100 audio DÀI và SẠCH nhất
  4. Chạy qua style_encoder → thu 100 tensor [1, 128]
  5. Tính trung bình → Mean Style Vector [1, 128]
  6. Lưu ra ngan_mean_style.pt

Đầu vào : - Checkpoint StyleTTS2 (Stage 3)
           - ngan_train_list.txt (wav_path|phoneme|speaker_id)
           - Config YAML của StyleTTS2 (để khởi tạo model)

Đầu ra  : ngan_mean_style.pt

Chạy lệnh:
    python create_mean_style.py \
        --checkpoint "Models/NganFinetune/best_model.pth" \
        --filelist   "data_pipeline/prepare_ngan/output/ngan_train_list.txt" \
        --config     "config/config_stage3.yaml"

    python create_mean_style.py --config config.yaml
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
from typing import Optional, List

import yaml
import numpy as np
import torch
import torchaudio
from dotenv import load_dotenv

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
class MeanStyleConfig:
    """Cấu hình cho trích xuất mean style vector."""

    # --- Đường dẫn bắt buộc ---
    # Checkpoint StyleTTS2 đã train xong Stage 3
    checkpoint_path: str = ""

    # Config YAML của StyleTTS2 (để khởi tạo đúng kiến trúc model)
    styletts2_config: str = ""

    # Filelist giọng Bác Ngạn (wav_path|phoneme|speaker_id)
    filelist_path: str = ""

    # Đường dẫn repo gốc StyleTTS2 (để import modules)
    styletts2_root: str = ""

    # --- Output ---
    output_file: str = "ngan_mean_style.pt"

    # --- Sampling ---
    # Số audio dùng để tính mean (chọn N audio dài nhất)
    n_samples: int = 100

    # Thời lượng tối thiểu của audio (giây) để được xét
    min_duration_s: float = 5.0

    # Random seed
    random_seed: int = 42

    # --- Hardware ---
    device: str = "cuda"

    # --- Audio ---
    sample_rate: int = 24000

    # Work dir
    work_dir: str = "./workdir"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "MeanStyleConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        ms = full_config.get("mean_style", {})
        paths = full_config.get("paths", {})

        config = cls()
        config.checkpoint_path = ms.get("checkpoint_path", config.checkpoint_path)
        config.styletts2_config = ms.get("styletts2_config", config.styletts2_config)
        config.filelist_path = ms.get("filelist_path", config.filelist_path)
        config.styletts2_root = ms.get("styletts2_root",
                                       paths.get("styletts2_root", config.styletts2_root))
        config.output_file = ms.get("output_file", config.output_file)
        config.n_samples = ms.get("n_samples", config.n_samples)
        config.min_duration_s = ms.get("min_duration_s", config.min_duration_s)
        config.random_seed = ms.get("random_seed", config.random_seed)
        config.device = ms.get("device", config.device)
        config.work_dir = ms.get("work_dir", paths.get("work_dir", config.work_dir))

        return config

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "create_mean_style.log"

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
    return logging.getLogger("mean_style")

# FILELIST PARSING & AUDIO SELECTION
def parse_filelist(filelist_path: str) -> List[dict]:
    """
    Đọc filelist (wav_path|phoneme|speaker_id), trả về list of dicts.
    """
    records = []
    with open(filelist_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) >= 1:
                records.append({
                    "wav_path": parts[0].strip(),
                    "phoneme": parts[1].strip() if len(parts) > 1 else "",
                    "speaker_id": int(parts[2].strip()) if len(parts) > 2 else 0,
                })
    return records

def get_audio_duration(wav_path: str, sample_rate: int = 24000) -> float:
    """Tính thời lượng audio (giây) mà không cần load toàn bộ."""
    try:
        info = torchaudio.info(wav_path)
        return info.num_frames / info.sample_rate
    except Exception:
        # Fallback: load và đo
        try:
            waveform, sr = torchaudio.load(wav_path)
            return waveform.shape[1] / sr
        except Exception:
            return 0.0

def select_best_samples(
    records: List[dict],
    n_samples: int,
    min_duration_s: float,
    random_seed: int,
    logger: logging.Logger,
) -> List[dict]:
    """
    Chọn N audio dài nhất & sạch nhất từ filelist.
    Ưu tiên audio dài vì style embedding ổn định hơn.
    """
    logger.info(f"Đang quét thời lượng {len(records):,} audio files...")

    # Tính duration cho từng file
    valid_records = []
    missing = 0

    for rec in records:
        wav_path = rec["wav_path"]
        if not Path(wav_path).exists():
            missing += 1
            continue

        duration = get_audio_duration(wav_path)
        if duration >= min_duration_s:
            rec["duration"] = duration
            valid_records.append(rec)

    logger.info(f"  Tổng records      : {len(records):,}")
    logger.info(f"  Wav không tồn tại : {missing}")
    logger.info(f"  Đủ dài (>= {min_duration_s}s) : {len(valid_records):,}")

    if not valid_records:
        logger.error("Không có audio nào đủ dài!")
        return []

    # Sắp xếp theo duration giảm dần
    valid_records.sort(key=lambda r: r["duration"], reverse=True)

    # Chọn top N (hoặc ít hơn nếu không đủ)
    n_select = min(n_samples, len(valid_records))

    # Nếu có nhiều hơn n_samples, random chọn từ top 2*n_samples
    # (để tránh luôn chọn cùng 1 bộ, tăng diversity)
    pool_size = min(n_samples * 2, len(valid_records))
    pool = valid_records[:pool_size]

    random.seed(random_seed)
    selected = random.sample(pool, n_select)

    durations = [r["duration"] for r in selected]
    logger.info(f"  Đã chọn {n_select} samples:")
    logger.info(f"    Duration min : {min(durations):.2f}s")
    logger.info(f"    Duration max : {max(durations):.2f}s")
    logger.info(f"    Duration avg : {sum(durations) / len(durations):.2f}s")

    return selected

# MODEL LOADING
def load_style_encoder(
    checkpoint_path: str,
    styletts2_config_path: str,
    styletts2_root: str,
    device: torch.device,
    logger: logging.Logger,
):
    """
    Load StyleTTS2 model và trích xuất style_encoder.

    Returns:
        (model, style_encoder, config_dict)
    """
    # Thêm repo gốc vào sys.path để import modules
    root = Path(styletts2_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        logger.info(f"Thêm vào sys.path: {root}")

    # Load config
    with open(styletts2_config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info(f"Loaded StyleTTS2 config: {styletts2_config_path}")

    # Import model builder từ repo gốc
    try:
        from models import build_model
        logger.info("Imported build_model từ StyleTTS2")
    except ImportError as e:
        logger.error(f"Không thể import 'models.build_model': {e}")
        logger.error(f"Kiểm tra lại styletts2_root: {styletts2_root}")
        raise

    # Build model
    model_params = config.get("model_params", {})

    # Khởi tạo tất cả modules
    model = build_model(model_params, stage="second")  # full model

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Xử lý các format checkpoint khác nhau
    if "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict vào từng module
    for key in model:
        if key in state_dict and hasattr(model[key], "load_state_dict"):
            try:
                model[key].load_state_dict(state_dict[key], strict=False)
                logger.info(f"  Loaded module: {key}")
            except Exception as e:
                logger.warning(f"  Skip module {key}: {e}")

    # Chuyển sang eval mode và device
    for key in model:
        if hasattr(model[key], "eval"):
            model[key].eval()
            model[key].to(device)

    # Trích xuất style_encoder
    style_encoder = model.get("style_encoder", None)
    if style_encoder is None:
        logger.error("Không tìm thấy 'style_encoder' trong model!")
        logger.error(f"Các module có sẵn: {list(model.keys())}")
        raise KeyError("style_encoder not found in model")

    logger.info("Style encoder loaded thành công!")

    return model, style_encoder, config

# MEL SPECTROGRAM
def compute_mel(
    wav_path: str,
    config: dict,
    device: torch.device,
) -> torch.Tensor:
    """
    Load audio và tính mel spectrogram theo đúng config StyleTTS2.

    Returns:
        Mel tensor [1, n_mels, T]
    """
    preprocess = config.get("preprocess_params", {})
    sr = preprocess.get("sr", 24000)
    spect = preprocess.get("spect_params", {})
    n_fft = spect.get("n_fft", 2048)
    win_length = spect.get("win_length", 1200)
    hop_length = spect.get("hop_length", 300)
    n_mels = config.get("model_params", {}).get("n_mels", 80)

    # Load audio
    waveform, orig_sr = torchaudio.load(wav_path)

    # Resample nếu cần
    if orig_sr != sr:
        resampler = torchaudio.transforms.Resample(orig_sr, sr)
        waveform = resampler(waveform)

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Mel spectrogram
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        n_mels=n_mels,
        norm="slaney",
        mel_scale="slaney",
    )

    mel = mel_transform(waveform)

    # Log scale
    mel = torch.log(torch.clamp(mel, min=1e-5))

    return mel.to(device)

# CORE LOGIC
@torch.no_grad()
def extract_mean_style(config: MeanStyleConfig, logger: logging.Logger):
    """
    Quy trình chính: Load model → chọn audio → extract style → mean → save.
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # --- Parse filelist ---
    logger.info(f"Đọc filelist: {config.filelist_path}")
    records = parse_filelist(config.filelist_path)
    logger.info(f"  Tổng records: {len(records):,}")

    # --- Chọn audio ---
    selected = select_best_samples(
        records,
        config.n_samples,
        config.min_duration_s,
        config.random_seed,
        logger,
    )

    if not selected:
        logger.error("Không chọn được sample nào! Kiểm tra filelist và wav files.")
        return

    # --- Load model ---
    logger.info("")
    logger.info("Loading StyleTTS2 model...")
    model, style_encoder, styletts2_config = load_style_encoder(
        config.checkpoint_path,
        config.styletts2_config,
        config.styletts2_root,
        device,
        logger,
    )

    # --- Extract style vectors ---
    logger.info("")
    logger.info(f"Trích xuất style vectors từ {len(selected)} samples...")

    style_vectors = []
    errors = 0
    start_time = time.time()

    for i, rec in enumerate(selected):
        wav_path = rec["wav_path"]
        try:
            # Tính mel
            mel = compute_mel(wav_path, styletts2_config, device)

            # Extract style
            # style_encoder thường nhận mel [1, n_mels, T] → output [1, style_dim]
            style_vec = style_encoder(mel)

            # Đảm bảo shape [1, style_dim]
            if style_vec.dim() == 1:
                style_vec = style_vec.unsqueeze(0)

            style_vectors.append(style_vec.cpu())

            if (i + 1) % 20 == 0:
                logger.info(f"  Processed {i + 1}/{len(selected)}...")

        except Exception as e:
            errors += 1
            if errors <= 10:
                logger.warning(f"  Lỗi xử lý {Path(wav_path).name}: {e}")

    elapsed = time.time() - start_time

    if not style_vectors:
        logger.error("Không trích xuất được style vector nào!")
        return

    logger.info(f"  Thành công: {len(style_vectors)}/{len(selected)} | Lỗi: {errors}")

    # --- Tính Mean ---
    all_styles = torch.cat(style_vectors, dim=0)  # [N, style_dim]
    mean_style = torch.mean(all_styles, dim=0, keepdim=True)  # [1, style_dim]

    logger.info(f"  All styles shape   : {all_styles.shape}")
    logger.info(f"  Mean style shape   : {mean_style.shape}")
    logger.info(f"  Mean style norm    : {mean_style.norm().item():.4f}")
    logger.info(f"  Mean style min/max : [{mean_style.min().item():.4f}, {mean_style.max().item():.4f}]")

    # --- Tính variance (kiểm tra tính nhất quán) ---
    variance = torch.var(all_styles, dim=0).mean().item()
    logger.info(f"  Style variance     : {variance:.6f}")
    if variance > 1.0:
        logger.warning("  Variance cao! Giọng Bác Ngạn có thể không nhất quán.")
        logger.warning("  Cân nhắc tăng n_samples hoặc lọc kỹ hơn.")

    # --- Lưu file ---
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(mean_style, str(output_path))

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ TRÍCH XUẤT MEAN STYLE VECTOR")
    logger.info("=" * 60)
    logger.info(f"  Thời gian       : {elapsed:.1f}s")
    logger.info(f"  Samples dùng    : {len(style_vectors)}")
    logger.info(f"  Style dim       : {mean_style.shape[1]}")
    logger.info(f"  Variance        : {variance:.6f}")
    logger.info(f"  Output file     : {output_path}")
    logger.info(f"  File size       : {output_path.stat().st_size / 1024:.1f} KB")
    logger.info("")
    logger.info("  Cách sử dụng trong tts_generator.py:")
    logger.info(f"    ref_s = torch.load('{output_path}')")
    logger.info("    wav = model.inference(text=phonemes, ref_s=ref_s, alpha=0.3, beta=0.7)")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Create Mean Style — Trích xuất Mean Style Vector giọng Bác Ngạn"
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override đường dẫn checkpoint StyleTTS2")
    parser.add_argument("--filelist", type=str, default=None,
                        help="Override đường dẫn filelist Bác Ngạn")
    parser.add_argument("--styletts2-config", type=str, default=None,
                        help="Override đường dẫn config YAML của StyleTTS2")
    parser.add_argument("--styletts2-root", type=str, default=None,
                        help="Override đường dẫn repo gốc StyleTTS2")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Override đường dẫn output .pt")
    parser.add_argument("--n-samples", type=int, default=None,
                        help="Override số lượng audio samples")
    args = parser.parse_args()

    # Load .env
    env_candidates = [Path(".env"), Path("../.env")]
    for ep in env_candidates:
        if ep.exists():
            load_dotenv(str(ep))
            break

    # Load config
    config_path = Path(args.config)
    if config_path.exists():
        config = MeanStyleConfig.from_yaml(str(config_path))
    else:
        config = MeanStyleConfig()

    # Infer paths
    if not config.styletts2_root:
        config.styletts2_root = str(
            (Path(__file__).parent.parent / "StyleTTS2").resolve()
        )

    # Override từ CLI
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.filelist:
        config.filelist_path = args.filelist
    if args.styletts2_config:
        config.styletts2_config = args.styletts2_config
    if args.styletts2_root:
        config.styletts2_root = args.styletts2_root
    if args.output:
        config.output_file = args.output
    if args.n_samples:
        config.n_samples = args.n_samples

    # Validate
    errors = []
    if not config.checkpoint_path:
        errors.append("Chưa chỉ định --checkpoint (checkpoint StyleTTS2 Stage 3)")
    elif not Path(config.checkpoint_path).exists():
        errors.append(f"Checkpoint không tồn tại: {config.checkpoint_path}")

    if not config.filelist_path:
        errors.append("Chưa chỉ định --filelist (ngan_train_list.txt)")
    elif not Path(config.filelist_path).exists():
        errors.append(f"Filelist không tồn tại: {config.filelist_path}")

    if not config.styletts2_config:
        errors.append("Chưa chỉ định --styletts2-config (config_stage3.yaml)")

    if errors:
        print("[LỖI] Thiếu tham số bắt buộc:")
        for e in errors:
            print(f"  - {e}")
        print("\nVí dụ:")
        print('  python create_mean_style.py \\')
        print('      --checkpoint "Models/NganFinetune/epoch_00050.pth" \\')
        print('      --filelist   "data_pipeline/prepare_ngan/output/ngan_train_list.txt" \\')
        print('      --styletts2-config "config/_processed/config_stage3_processed.yaml"')
        sys.exit(1)

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  CREATE MEAN STYLE — TRÍCH XUẤT GIỌNG BÁC NGẠN")
    logger.info("=" * 60)
    logger.info(f"Config           : {config_path}")
    logger.info(f"Checkpoint       : {config.checkpoint_path}")
    logger.info(f"Filelist         : {config.filelist_path}")
    logger.info(f"StyleTTS2 config : {config.styletts2_config}")
    logger.info(f"StyleTTS2 root   : {config.styletts2_root}")
    logger.info(f"N samples        : {config.n_samples}")
    logger.info(f"Min duration     : {config.min_duration_s}s")
    logger.info(f"Output           : {config.output_file}")
    logger.info(f"Device           : {config.device}")

    # Run
    try:
        extract_mean_style(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()