"""
=============================================================================
  TTS GENERATOR — Phase 2: Thu âm & Hậu kỳ (Module TTS với StyleTTS2)
=============================================================================
Mục tiêu: Đọc script.json (từ Phase 1 NLP), chuyển text → phoneme,
          gọi StyleTTS2 sinh âm thanh, chèn khoảng lặng (np.zeros) xen kẽ,
          nối tất cả thành file audiobook .wav cuối cùng.

Quy trình:
  1. Load model StyleTTS2 (checkpoint Stage 3)
  2. Load phoneme_vocab.json + ngan_mean_style.pt
  3. Đọc script.json
  4. Vòng lặp: text → phonemize → TTS inference → append audio
  5. Chèn silence (np.zeros) theo pause_after_ms
  6. Nối tất cả → xuất output_ghost_story.wav

Yêu cầu: vLLM server ĐÃ TẮT (VRAM trống cho StyleTTS2)

Chạy lệnh:
    python tts_generator.py \
        --script    "script.json" \
        --checkpoint "Models/NganFinetune/epoch_00050.pth" \
        --style     "ngan_mean_style.pt" \
        --config    "config/_processed/config_stage3_processed.yaml"

    python tts_generator.py --config config.yaml
=============================================================================
"""

import os
import sys
import json
import time
import logging
import argparse
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, List

import yaml
import numpy as np
import torch
import soundfile as sf
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

# KHẮC PHỤC LỖI "WinError 193" — vinorm monkey-patch
import vinorm

def _mock_tts_norm(text, *args, **kwargs):
    return str(text).lower().strip()

vinorm.TTSnorm = _mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

import viphoneme
viphoneme.TTSnorm = _mock_tts_norm

from viphoneme import vi2IPA_split

# CONFIGURATION
@dataclass
class TTSConfig:
    """Cấu hình cho TTS Generator."""

    # --- Đường dẫn bắt buộc ---
    script_file: str = "script.json"
    checkpoint_path: str = ""
    styletts2_config: str = ""
    style_vector_path: str = "ngan_mean_style.pt"
    styletts2_root: str = ""

    # --- Vocab ---
    vocab_file: str = ""

    # --- Output ---
    output_file: str = "output_ghost_story.wav"
    output_dir: str = "./output"

    # --- Inference params ---
    # alpha: weight cho style (0=neutral, 1=full style). 0.3 = nhẹ nhàng
    alpha: float = 0.3
    # beta: weight cho prosody prediction. 0.7 = dùng nhiều prosody predictor
    beta: float = 0.7
    # diffusion_steps: số bước diffusion (cao hơn = chất lượng hơn nhưng chậm hơn)
    diffusion_steps: int = 5
    # embedding_scale: scale cho classifier-free guidance
    embedding_scale: float = 1.0

    # --- Audio ---
    sample_rate: int = 24000

    # --- Hardware ---
    device: str = "cuda"

    # --- Xuất audio từng câu (debug) ---
    save_individual: bool = False
    individual_dir: str = "individual_wavs"

    # Work dir
    work_dir: str = "./workdir"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "TTSConfig":
        """Load config từ file YAML."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        tts = full_config.get("tts_generator", {})
        paths = full_config.get("paths", {})

        config = cls()
        config.script_file = tts.get("script_file", config.script_file)
        config.checkpoint_path = tts.get("checkpoint_path", config.checkpoint_path)
        config.styletts2_config = tts.get("styletts2_config", config.styletts2_config)
        config.style_vector_path = tts.get("style_vector_path", config.style_vector_path)
        config.styletts2_root = tts.get("styletts2_root",
                                        paths.get("styletts2_root", config.styletts2_root))
        config.vocab_file = tts.get("vocab_file", paths.get("vocab_file", config.vocab_file))
        config.output_file = tts.get("output_file", config.output_file)
        config.output_dir = tts.get("output_dir", config.output_dir)
        config.alpha = tts.get("alpha", config.alpha)
        config.beta = tts.get("beta", config.beta)
        config.diffusion_steps = tts.get("diffusion_steps", config.diffusion_steps)
        config.embedding_scale = tts.get("embedding_scale", config.embedding_scale)
        config.device = tts.get("device", config.device)
        config.save_individual = tts.get("save_individual", config.save_individual)
        config.work_dir = tts.get("work_dir", paths.get("work_dir", config.work_dir))

        return config

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "tts_generator.log"

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
    return logging.getLogger("tts_generator")

# PHONEMIZATION
def clean_text_for_tts(text: str) -> str:
    """
    Làm sạch text trước khi phonemize cho TTS inference.
    Nhẹ hơn clean cho training — giữ lại dấu câu để prosody tự nhiên.
    """
    # Xóa ký tự đặc biệt rườm rà
    text = re.sub(r'["""\'\(\)\[\]\{\}<>«»—–\\\|]', '', text)

    # Chuẩn hóa dấu câu liên tiếp
    text = re.sub(r'\.{2,}', '.', text)
    text = re.sub(r'!{2,}', '!', text)
    text = re.sub(r'\?{2,}', '?', text)

    # Chuyển số thành chữ
    digit_map = {
        "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
        "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
    }
    result = []
    for char in text:
        if char.isdigit():
            result.append(digit_map.get(char, char))
            result.append(" ")
        else:
            result.append(char)
    text = "".join(result)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()

    return text

def text_to_phoneme(text: str) -> str:
    """Chuyển text tiếng Việt → chuỗi phoneme IPA."""
    try:
        phonemes = vi2IPA_split(text, " ")
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes
    except Exception as e:
        return f"[ERROR] {str(e)}"

def phonemes_to_ids(phoneme_str: str, vocab: dict) -> List[int]:
    """
    Chuyển chuỗi phoneme thành list token IDs dựa trên vocab.
    """
    char_to_id = vocab.get("char_to_id", {})
    unk_id = char_to_id.get("<unk>", 1)

    ids = []
    for char in phoneme_str:
        ids.append(char_to_id.get(char, unk_id))

    return ids

# MODEL LOADING
def load_styletts2_model(
    checkpoint_path: str,
    styletts2_config_path: str,
    styletts2_root: str,
    device: torch.device,
    logger: logging.Logger,
) -> dict:
    """
    Load toàn bộ StyleTTS2 model cho inference.
    """
    root = Path(styletts2_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
        logger.info(f"Thêm vào sys.path: {root}")

    # Load config
    with open(styletts2_config_path, "r", encoding="utf-8") as f:
        model_config = yaml.safe_load(f)

    # Import từ repo gốc
    try:
        from models import build_model
        logger.info("Imported build_model từ StyleTTS2")
    except ImportError as e:
        logger.error(f"Không thể import 'models.build_model': {e}")
        raise

    # Build model
    model_params = model_config.get("model_params", {})
    model = build_model(model_params, stage="second")

    # Load checkpoint
    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    if "net" in checkpoint:
        state_dict = checkpoint["net"]
    elif "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    # Load state dict
    for key in model:
        if key in state_dict and hasattr(model[key], "load_state_dict"):
            try:
                model[key].load_state_dict(state_dict[key], strict=False)
            except Exception as e:
                logger.warning(f"  Skip module {key}: {e}")

    # Eval mode + device
    for key in model:
        if hasattr(model[key], "eval"):
            model[key].eval()
            model[key].to(device)

    logger.info("StyleTTS2 model loaded thành công!")
    return model, model_config

def load_inference_function(styletts2_root: str, logger: logging.Logger):
    """
    Import hàm inference từ repo gốc StyleTTS2.
    StyleTTS2 thường có inference function trong inference.py hoặc trong demo notebook.
    """
    root = Path(styletts2_root)
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    try:
        # Thử import từ inference module
        from inference import inference as styletts2_inference
        logger.info("Imported inference function từ inference.py")
        return styletts2_inference
    except ImportError:
        pass

    try:
        # Thử import từ utils
        from utils import inference as styletts2_inference
        logger.info("Imported inference function từ utils.py")
        return styletts2_inference
    except ImportError:
        pass

    # Nếu không tìm thấy, trả về None → sẽ dùng manual inference
    logger.warning("Không tìm thấy inference function sẵn có trong repo gốc.")
    logger.warning("Sẽ dùng manual inference pipeline.")
    return None

# INFERENCE WRAPPER
@torch.no_grad()
def synthesize_single(
    text: str,
    model: dict,
    mean_style: torch.Tensor,
    vocab: dict,
    device: torch.device,
    inference_fn=None,
    alpha: float = 0.3,
    beta: float = 0.7,
    diffusion_steps: int = 5,
    embedding_scale: float = 1.0,
    sample_rate: int = 24000,
) -> Optional[np.ndarray]:
    """
    Sinh âm thanh cho 1 câu text.

    Flow:
      text → clean → phonemize → token IDs → model inference → numpy audio

    Returns:
        numpy array 1D (float32) chứa waveform, hoặc None nếu thất bại.
    """
    # Clean text
    cleaned = clean_text_for_tts(text)
    if not cleaned:
        return None

    # Phonemize
    phoneme_str = text_to_phoneme(cleaned)
    if phoneme_str.startswith("[ERROR]") or not phoneme_str:
        return None

    # Token IDs
    token_ids = phonemes_to_ids(phoneme_str, vocab)
    if not token_ids:
        return None

    # Tensor
    tokens = torch.LongTensor([token_ids]).to(device)
    token_lengths = torch.LongTensor([len(token_ids)]).to(device)

    # Style vector
    ref_s = mean_style.to(device)

    # Inference
    try:
        if inference_fn is not None:
            # Dùng inference function của repo gốc
            wav = inference_fn(
                tokens,
                token_lengths,
                ref_s,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale,
            )
        else:
            # Manual inference (fallback)
            # Đây là skeleton — cần điều chỉnh theo API chính xác của build_model
            wav = _manual_inference(
                model, tokens, token_lengths, ref_s,
                alpha, beta, diffusion_steps, embedding_scale,
            )

        # Chuyển sang numpy
        if isinstance(wav, torch.Tensor):
            wav = wav.cpu().numpy()

        # Flatten
        wav = wav.squeeze().astype(np.float32)

        # Kiểm tra hợp lệ
        if len(wav) == 0 or np.all(wav == 0):
            return None

        return wav

    except Exception as e:
        return None

def _manual_inference(
    model: dict,
    tokens: torch.Tensor,
    token_lengths: torch.Tensor,
    ref_s: torch.Tensor,
    alpha: float,
    beta: float,
    diffusion_steps: int,
    embedding_scale: float,
) -> torch.Tensor:
    """
    Manual inference pipeline khi không có inference function sẵn.
    Skeleton — cần điều chỉnh dựa trên API thực tế của StyleTTS2.

    Lưu ý: Đây là simplified version. Nếu repo gốc cung cấp inference.py,
    hãy ưu tiên dùng inference function đó.
    """
    # Encode text
    text_encoder = model.get("text_encoder", None)
    style_encoder = model.get("style_encoder", None)
    decoder = model.get("decoder", None)
    predictor = model.get("predictor", None)
    text_aligner = model.get("text_aligner", None)

    if text_encoder is None or decoder is None:
        raise RuntimeError(
            "Model thiếu modules cần thiết cho inference. "
            "Hãy kiểm tra build_model() hoặc dùng inference.py từ repo gốc."
        )

    # Encode
    t_en = text_encoder(tokens, token_lengths, None)

    # Style
    s = ref_s

    # Duration / Alignment prediction
    if predictor is not None:
        d = predictor(t_en, s, token_lengths, alpha=alpha, beta=beta)
    else:
        raise RuntimeError("Predictor module not found")

    # Decode
    wav = decoder(t_en, d, s)

    return wav

# SILENCE GENERATION
def create_silence(duration_ms: int, sample_rate: int = 24000) -> np.ndarray:
    """
    Tạo mảng im lặng tuyệt đối (np.zeros).
    Khoảng lặng sạch sẽ, không noise — chuẩn phim kinh dị.
    """
    num_samples = int((duration_ms / 1000.0) * sample_rate)
    return np.zeros(num_samples, dtype=np.float32)

# CORE LOGIC
def generate_audiobook(config: TTSConfig, logger: logging.Logger):
    """
    Quy trình chính Phase 2:
    1. Load model + style + vocab
    2. Đọc script.json
    3. Vòng lặp: text → phoneme → TTS → silence → append
    4. Nối tất cả → xuất .wav
    """
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        vram_total = torch.cuda.get_device_properties(0).total_mem / 1e9
        vram_free = torch.cuda.mem_get_info()[0] / 1e9
        logger.info(f"VRAM: {vram_free:.1f} / {vram_total:.1f} GB free")

    # --- Load script.json ---
    script_path = Path(config.script_file)
    if not script_path.exists():
        logger.error(f"Không tìm thấy script.json: {script_path}")
        logger.error("Hãy chạy nlp_generator.py (Phase 1) trước!")
        return

    with open(script_path, "r", encoding="utf-8") as f:
        script_items = json.load(f)

    logger.info(f"Loaded script: {len(script_items)} items từ {script_path.name}")

    # --- Load vocab ---
    vocab = {}
    if config.vocab_file and Path(config.vocab_file).exists():
        with open(config.vocab_file, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        logger.info(f"Loaded vocab: {vocab.get('n_token', '?')} tokens")
    else:
        logger.warning("Vocab file không được chỉ định — phoneme IDs có thể không chính xác!")

    # --- Load mean style vector ---
    style_path = Path(config.style_vector_path)
    if not style_path.exists():
        logger.error(f"Không tìm thấy style vector: {style_path}")
        logger.error("Hãy chạy create_mean_style.py trước!")
        return

    mean_style = torch.load(str(style_path), map_location="cpu")
    logger.info(f"Loaded mean style: shape={mean_style.shape}")

    # --- Load model ---
    logger.info("")
    logger.info("Loading StyleTTS2 model...")
    model, model_config = load_styletts2_model(
        config.checkpoint_path,
        config.styletts2_config,
        config.styletts2_root,
        device,
        logger,
    )

    # --- Load inference function ---
    inference_fn = load_inference_function(config.styletts2_root, logger)

    # --- Chuẩn bị output ---
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if config.save_individual:
        individual_dir = output_dir / config.individual_dir
        individual_dir.mkdir(parents=True, exist_ok=True)

    # --- VÒNG LẶP SINH ÂM THANH ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  BẮT ĐẦU SINH ÂM THANH")
    logger.info("=" * 60)

    audio_chunks = []
    stats = {
        "total": len(script_items),
        "success": 0,
        "failed": 0,
        "total_audio_s": 0.0,
        "total_silence_s": 0.0,
    }

    start_time = time.time()

    for item in script_items:
        item_id = item.get("id", 0)
        role = item.get("role", "narrator")
        text = item.get("text", "").strip()
        pause_ms = item.get("pause_after_ms", 500)

        if not text:
            stats["failed"] += 1
            continue

        # --- Sinh âm thanh ---
        wav_array = synthesize_single(
            text=text,
            model=model,
            mean_style=mean_style,
            vocab=vocab,
            device=device,
            inference_fn=inference_fn,
            alpha=config.alpha,
            beta=config.beta,
            diffusion_steps=config.diffusion_steps,
            embedding_scale=config.embedding_scale,
            sample_rate=config.sample_rate,
        )

        if wav_array is None:
            stats["failed"] += 1
            logger.warning(f"  [{item_id:3d}] FAILED: {text[:50]}...")
            continue

        # Append audio
        audio_chunks.append(wav_array)
        audio_duration = len(wav_array) / config.sample_rate
        stats["total_audio_s"] += audio_duration
        stats["success"] += 1

        # Lưu individual (debug)
        if config.save_individual:
            ind_path = individual_dir / f"item_{item_id:04d}.wav"
            sf.write(str(ind_path), wav_array, config.sample_rate, subtype="PCM_16")

        # --- Chèn khoảng lặng ---
        silence = create_silence(pause_ms, config.sample_rate)
        audio_chunks.append(silence)
        stats["total_silence_s"] += pause_ms / 1000.0

        # Log tiến độ
        if item_id % 10 == 0 or item_id <= 5:
            text_preview = text[:40] + "..." if len(text) > 40 else text
            logger.info(
                f"  [{item_id:3d}/{stats['total']}] "
                f"[{role:9s}] {text_preview} | "
                f"audio={audio_duration:.2f}s | pause={pause_ms}ms"
            )

    elapsed = time.time() - start_time

    # --- Kiểm tra kết quả ---
    if not audio_chunks:
        logger.error("Không sinh được audio nào! Kiểm tra model và config.")
        return

    # --- Nối tất cả ---
    logger.info("")
    logger.info("Nối tất cả audio chunks...")
    final_wav = np.concatenate(audio_chunks)

    # --- Normalize final output ---
    peak = np.abs(final_wav).max()
    if peak > 0:
        final_wav = final_wav / peak * 0.95  # Headroom 5%

    # --- Lưu file ---
    output_path = output_dir / config.output_file
    sf.write(str(output_path), final_wav, config.sample_rate, subtype="PCM_16")

    total_duration = len(final_wav) / config.sample_rate

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ TTS GENERATOR — PHASE 2")
    logger.info("=" * 60)
    logger.info(f"  Thời gian render   : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Tổng items         : {stats['total']}")
    logger.info(f"  Thành công         : {stats['success']}")
    logger.info(f"  Thất bại           : {stats['failed']}")
    logger.info("")
    logger.info(f"  Tổng audio speech  : {stats['total_audio_s']:.1f}s ({stats['total_audio_s'] / 60:.1f} phút)")
    logger.info(f"  Tổng silence       : {stats['total_silence_s']:.1f}s ({stats['total_silence_s'] / 60:.1f} phút)")
    logger.info(f"  Tổng audiobook     : {total_duration:.1f}s ({total_duration / 60:.1f} phút)")
    logger.info(f"  Tỷ lệ silence      : {stats['total_silence_s'] / total_duration * 100:.1f}%")
    logger.info("")
    logger.info(f"  Output file        : {output_path}")
    logger.info(f"  File size          : {output_path.stat().st_size / 1e6:.1f} MB")
    logger.info(f"  Sample rate        : {config.sample_rate} Hz")
    logger.info(f"  Format             : WAV 16-bit PCM")

    if config.save_individual:
        logger.info(f"  Individual wavs    : {individual_dir}")

    logger.info("")
    logger.info("=" * 60)
    logger.info("  AUDIOBOOK TRUYỆN MA ĐÃ HOÀN TẤT!")
    logger.info("=" * 60)
    logger.info(f"  File: {output_path}")
    logger.info("  Mở bằng bất kỳ audio player nào để nghe.")
    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="TTS Generator — Phase 2: Sinh audiobook truyện ma từ script.json"
    )
    parser.add_argument("--config", "-c", type=str, default="config.yaml")
    parser.add_argument("--script", type=str, default=None,
                        help="Override đường dẫn script.json")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Override checkpoint StyleTTS2")
    parser.add_argument("--style", type=str, default=None,
                        help="Override đường dẫn ngan_mean_style.pt")
    parser.add_argument("--styletts2-config", type=str, default=None,
                        help="Override config YAML của StyleTTS2")
    parser.add_argument("--styletts2-root", type=str, default=None,
                        help="Override đường dẫn repo gốc StyleTTS2")
    parser.add_argument("--vocab", type=str, default=None,
                        help="Override đường dẫn phoneme_vocab.json")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Override tên file output")
    parser.add_argument("--alpha", type=float, default=None,
                        help="Override alpha (style weight)")
    parser.add_argument("--beta", type=float, default=None,
                        help="Override beta (prosody weight)")
    parser.add_argument("--save-individual", action="store_true",
                        help="Lưu audio từng câu riêng (debug)")
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
        config = TTSConfig.from_yaml(str(config_path))
    else:
        config = TTSConfig()

    # Infer styletts2_root
    if not config.styletts2_root:
        config.styletts2_root = str(
            (Path(__file__).parent.parent / "StyleTTS2").resolve()
        )

    # Override từ CLI
    if args.script:
        config.script_file = args.script
    if args.checkpoint:
        config.checkpoint_path = args.checkpoint
    if args.style:
        config.style_vector_path = args.style
    if args.styletts2_config:
        config.styletts2_config = args.styletts2_config
    if args.styletts2_root:
        config.styletts2_root = args.styletts2_root
    if args.vocab:
        config.vocab_file = args.vocab
    if args.output:
        config.output_file = args.output
    if args.alpha is not None:
        config.alpha = args.alpha
    if args.beta is not None:
        config.beta = args.beta
    if args.save_individual:
        config.save_individual = True

    # Validate
    errors = []
    if not config.script_file or not Path(config.script_file).exists():
        errors.append(f"Script file không tồn tại: {config.script_file}")
    if not config.checkpoint_path:
        errors.append("Chưa chỉ định --checkpoint")
    elif not Path(config.checkpoint_path).exists():
        errors.append(f"Checkpoint không tồn tại: {config.checkpoint_path}")
    if not config.styletts2_config:
        errors.append("Chưa chỉ định --styletts2-config")
    if not config.style_vector_path or not Path(config.style_vector_path).exists():
        errors.append(f"Style vector không tồn tại: {config.style_vector_path}")

    if errors:
        print("[LỖI] Thiếu tham số bắt buộc:")
        for e in errors:
            print(f"  - {e}")
        print("\nVí dụ:")
        print('  python tts_generator.py \\')
        print('      --script     "script.json" \\')
        print('      --checkpoint "Models/NganFinetune/epoch_00050.pth" \\')
        print('      --style      "ngan_mean_style.pt" \\')
        print('      --styletts2-config "config/_processed/config_stage3_processed.yaml" \\')
        print('      --vocab      "data_pipeline/prepare_vicoice/output/phoneme_vocab.json"')
        sys.exit(1)

    # Setup logging
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # Header
    logger.info("=" * 60)
    logger.info("  TTS GENERATOR — PHASE 2: SINH AUDIOBOOK TRUYỆN MA")
    logger.info("=" * 60)
    logger.info(f"Config          : {config_path}")
    logger.info(f"Script          : {config.script_file}")
    logger.info(f"Checkpoint      : {config.checkpoint_path}")
    logger.info(f"Style vector    : {config.style_vector_path}")
    logger.info(f"StyleTTS2 config: {config.styletts2_config}")
    logger.info(f"Vocab           : {config.vocab_file or '(không chỉ định)'}")
    logger.info(f"Alpha/Beta      : {config.alpha}/{config.beta}")
    logger.info(f"Diffusion steps : {config.diffusion_steps}")
    logger.info(f"Output          : {Path(config.output_dir) / config.output_file}")
    logger.info(f"Save individual : {config.save_individual}")
    logger.info(f"Device          : {config.device}")

    # Run
    try:
        generate_audiobook(config, logger)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()