"""
=============================================================================
  BƯỚC 2: EXTRACT AUDIO — Giải mã audio từ dataset ViVoice → file .wav
=============================================================================
Mục tiêu: Lặp qua từng sample trong dataset đã tải (Bước 1), giải mã
          audio bytes → resample về 24kHz mono 16-bit → lưu thành .wav.
          Đồng thời trích xuất cột text thô ra file riêng cho Bước 3.

Đầu vào : Dataset ViVoice đã cache tại work_dir/hf_cache/ (từ Bước 1)
Đầu ra  : - Thư mục chứa file .wav đã chuẩn hóa
           - File raw_texts.txt (index tương ứng với .wav)

Chạy lệnh:
    python step2_extract_audio.py
    python step2_extract_audio.py --config config.yaml
=============================================================================
"""

import io
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# [FIX] Tắt torchcodec TRƯỚC KHI import bất kỳ thứ gì từ datasets.
#   - datasets >= 2.x tự động dùng torchcodec làm audio backend nếu có cài.
#   - torchcodec yêu cầu FFmpeg "full-shared" DLL khớp đúng ABI version.
#   - Cài FFmpeg thông thường (kể cả full_build) vẫn có thể lệch ABI minor
#     version → WinError 127 "procedure not found".
#   Giải pháp triệt để: load dataset với decode=False, tự decode bằng
#   soundfile (không cần FFmpeg / torchcodec).
os.environ["DATASETS_AUDIO_DECODE_WITH_TORCHCODEC"] = "0"   # datasets ≥ 2.21
os.environ["HF_DATASETS_AUDIO_DECODE_WITH_TORCHCODEC"] = "0"  # alias an toàn

import yaml
import numpy as np
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

    # os.add_dll_directory giữ lại phòng trường hợp thư viện khác cần FFmpeg,
    # nhưng pipeline audio này không phụ thuộc vào nó nữa.
    try:
        os.add_dll_directory(r"D:\FFmpeg\ffmpeg-7.0.2-full_build-shared\bin")
    except (AttributeError, OSError):
        pass

# CONFIGURATION
@dataclass
class ExtractConfig:
    """Cấu hình cho bước giải mã và chuẩn hóa audio."""

    # Đường dẫn
    work_dir: str = "./workdir"
    output_dir: str = "./output"
    cache_subdir: str = "hf_cache"
    wav_subdir: str = "vivoice_clean_wavs"

    # Thông tin dataset (để load lại từ cache)
    dataset_name: str = "capleaf/viVoice"
    dataset_config: Optional[str] = None
    dataset_split: Optional[str] = None

    # Audio params
    target_sr: int = 24000
    channels: int = 1
    bit_depth: int = 16
    audio_format: str = "wav"

    # File text thô (output cho step3)
    raw_text_file: str = "raw_texts.txt"

    # Tùy chọn
    num_workers: int = 4
    skip_existing: bool = True

    # HF token
    hf_token: Optional[str] = None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ExtractConfig":
        """Load config từ file YAML chung của prepare_vivoice."""
        with open(yaml_path, "r", encoding="utf-8") as f:
            full_config = yaml.safe_load(f)

        paths = full_config.get("paths", {})
        step1 = full_config.get("step1_download", {})
        step2 = full_config.get("step2_extract", {})
        step3 = full_config.get("step3_phonemize", {})

        return cls(
            work_dir=paths.get("work_dir", cls.work_dir),
            output_dir=paths.get("output_dir", cls.output_dir),
            cache_subdir=step1.get("cache_subdir", cls.cache_subdir),
            dataset_name=step1.get("dataset_name", cls.dataset_name),
            dataset_config=step1.get("dataset_config", cls.dataset_config),
            dataset_split=step1.get("dataset_split", cls.dataset_split),
            wav_subdir=step2.get("wav_subdir", cls.wav_subdir),
            target_sr=step2.get("target_sr", cls.target_sr),
            channels=step2.get("channels", cls.channels),
            bit_depth=step2.get("bit_depth", cls.bit_depth),
            audio_format=step2.get("format", cls.audio_format),
            raw_text_file=step3.get("raw_text_file", cls.raw_text_file),
            num_workers=step2.get("num_workers", cls.num_workers),
            skip_existing=step2.get("skip_existing", cls.skip_existing),
        )

# LOGGING SETUP
def setup_logging(log_dir: Path) -> logging.Logger:
    """Thiết lập logging ghi ra console + file."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "step2_extract_audio.log"

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
    return logging.getLogger("step2_extract")

# UTILITY FUNCTIONS
def resample_audio(audio_array: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """
    Resample audio array về target_sr.
    Sử dụng librosa nếu cần resample, trả về nguyên bản nếu sr đã khớp.
    """
    if orig_sr == target_sr:
        return audio_array

    import librosa
    resampled = librosa.resample(
        audio_array.astype(np.float32),
        orig_sr=orig_sr,
        target_sr=target_sr,
    )
    return resampled

def to_mono(audio_array: np.ndarray) -> np.ndarray:
    """Chuyển audio về mono nếu đang là stereo/multi-channel."""
    if audio_array.ndim == 1:
        return audio_array
    # Trung bình các kênh
    return np.mean(audio_array, axis=0)

def normalize_audio(audio_array: np.ndarray) -> np.ndarray:
    """
    Normalize audio về khoảng [-1.0, 1.0].
    Tránh chia cho 0 nếu audio toàn silence.
    """
    peak = np.abs(audio_array).max()
    if peak > 0:
        return audio_array / peak
    return audio_array

def save_wav(audio_array: np.ndarray, path: Path, sr: int, bit_depth: int = 16):
    """Lưu audio array thành file .wav với bit depth chỉ định."""
    subtype_map = {
        16: "PCM_16",
        24: "PCM_24",
        32: "FLOAT",
    }
    subtype = subtype_map.get(bit_depth, "PCM_16")
    sf.write(str(path), audio_array, sr, subtype=subtype)

# [FIX] Hàm decode audio thủ công — không dùng torchcodec / FFmpeg
def decode_audio_from_sample(audio_info: dict, target_sr: int) -> tuple:
    """
    Giải mã audio từ dict trả về bởi datasets (decode=False).

    datasets trả về dict với một trong các dạng:
      1. {"array": np.ndarray, "sampling_rate": int}   ← đã decode sẵn (hiếm)
      2. {"bytes": bytes, "path": str|None}             ← raw bytes (phổ biến)
      3. {"bytes": None,  "path": str}                  ← path tới file local

    Trả về: (audio_array: np.ndarray, orig_sr: int)
    Raise : ValueError nếu không decode được.
    """
    # --- Trường hợp 1: đã decode sẵn ---
    if audio_info.get("array") is not None:
        return np.array(audio_info["array"], dtype=np.float32), int(audio_info.get("sampling_rate", target_sr))

    # --- Trường hợp 2: raw bytes ---
    raw_bytes = audio_info.get("bytes")
    if raw_bytes is not None:
        try:
            audio_array, orig_sr = sf.read(io.BytesIO(raw_bytes), dtype="float32", always_2d=False)
            return audio_array, orig_sr
        except Exception as e_sf:
            # Fallback: librosa (hỗ trợ thêm MP3 qua audioread nếu cài)
            try:
                import librosa
                audio_array, orig_sr = librosa.load(io.BytesIO(raw_bytes), sr=None, mono=False)
                return audio_array.astype(np.float32), orig_sr
            except Exception as e_lib:
                raise ValueError(
                    f"soundfile lỗi: {e_sf} | librosa lỗi: {e_lib}"
                ) from e_lib

    # --- Trường hợp 3: path tới file local ---
    file_path = audio_info.get("path")
    if file_path and os.path.isfile(file_path):
        try:
            audio_array, orig_sr = sf.read(file_path, dtype="float32", always_2d=False)
            return audio_array, orig_sr
        except Exception as e_sf:
            try:
                import librosa
                audio_array, orig_sr = librosa.load(file_path, sr=None, mono=False)
                return audio_array.astype(np.float32), orig_sr
            except Exception as e_lib:
                raise ValueError(
                    f"soundfile lỗi: {e_sf} | librosa lỗi: {e_lib}"
                ) from e_lib

    raise ValueError(
        f"audio_info không có 'array', 'bytes', hay 'path' hợp lệ. Keys: {list(audio_info.keys())}"
    )

# CORE LOGIC
def process_single_sample(
    sample: dict,
    index: int,
    wav_dir: Path,
    target_sr: int,
    bit_depth: int,
    skip_existing: bool,
) -> dict:
    """
    Xử lý 1 sample từ dataset: giải mã audio → resample → lưu .wav.

    Returns:
        dict với keys: "wav_path", "text", "success", "error"
    """
    result = {
        "wav_path": None,
        "text": "",
        "success": False,
        "error": None,
    }

    try:
        # --- Trích xuất text ---
        text = sample.get("text", "").strip()
        result["text"] = text

        if not text:
            result["error"] = "Text rỗng"
            return result

        # --- Tạo tên file .wav ---
        wav_filename = f"vivoice_{index:07d}.wav"
        wav_path = wav_dir / wav_filename
        result["wav_path"] = str(wav_path)

        # --- Skip nếu đã tồn tại ---
        if skip_existing and wav_path.exists():
            result["success"] = True
            return result

        # --- Lấy audio info ---
        audio_info = sample.get("audio", None)
        if audio_info is None:
            result["error"] = "Không có cột audio"
            return result

        if not isinstance(audio_info, dict):
            result["error"] = f"Format audio không hỗ trợ: {type(audio_info)}"
            return result

        # --- [FIX] Decode audio thủ công, không qua torchcodec ---
        audio_array, orig_sr = decode_audio_from_sample(audio_info, target_sr)

        # --- Chuyển về mono ---
        audio_array = to_mono(audio_array)

        # --- Resample ---
        audio_array = resample_audio(audio_array, orig_sr, target_sr)

        # --- Kiểm tra audio hợp lệ ---
        duration_s = len(audio_array) / target_sr
        if duration_s < 0.1:
            result["error"] = f"Audio quá ngắn: {duration_s:.3f}s"
            return result

        if np.all(audio_array == 0):
            result["error"] = "Audio toàn silence"
            return result

        # --- Normalize ---
        audio_array = normalize_audio(audio_array)

        # --- Lưu .wav ---
        save_wav(audio_array, wav_path, target_sr, bit_depth)
        result["success"] = True

    except Exception as e:
        result["error"] = str(e)

    return result

def extract_dataset(config: ExtractConfig, logger: logging.Logger):
    """
    Quy trình chính: Load dataset từ cache → xử lý từng sample → lưu .wav + text.
    """
    # --- Lazy import ---
    from datasets import load_dataset, Audio
    from tqdm import tqdm

    # --- Chuẩn bị thư mục ---
    wav_dir = Path(config.output_dir) / config.wav_subdir
    wav_dir.mkdir(parents=True, exist_ok=True)

    work_dir = Path(config.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)

    cache_dir = work_dir / config.cache_subdir

    # --- Load dataset từ cache ---
    logger.info("Đang load dataset từ cache...")
    logger.info(f"  Cache dir  : {cache_dir}")
    logger.info(f"  Dataset    : {config.dataset_name}")

    load_kwargs = {
        "path": config.dataset_name,
        "cache_dir": str(cache_dir),
    }

    if config.dataset_config:
        load_kwargs["name"] = config.dataset_config

    if config.dataset_split:
        load_kwargs["split"] = config.dataset_split

    if config.hf_token:
        load_kwargs["token"] = config.hf_token

    dataset = load_dataset(**load_kwargs)

    # --- [FIX] Tắt auto-decode audio để datasets không gọi torchcodec ---
    # Audio(decode=False) → datasets trả về raw bytes thay vì decoded array.
    # process_single_sample sẽ tự decode bằng soundfile.
    def _disable_audio_decode(ds):
        if "audio" in ds.features:
            return ds.cast_column("audio", Audio(decode=False))
        return ds

    # --- Xác định splits cần xử lý ---
    if hasattr(dataset, "keys"):
        splits = list(dataset.keys())
        logger.info(f"  Splits     : {splits}")
        for split_name in splits:
            dataset[split_name] = _disable_audio_decode(dataset[split_name])
    else:
        splits = ["train"]
        dataset = {"train": _disable_audio_decode(dataset)}

    logger.info("  Audio decode: soundfile (torchcodec bypassed)")

    # --- Xử lý từng split ---
    global_index = 0
    all_texts = []
    all_wav_paths = []

    stats = {
        "total": 0,
        "success": 0,
        "skipped_existing": 0,
        "errors": 0,
        "error_details": {},
    }

    start_time = time.time()

    for split_name in splits:
        split_data = dataset[split_name]
        split_size = len(split_data)
        logger.info(f"\nĐang xử lý split '{split_name}': {split_size:,} samples")

        for i in tqdm(range(split_size), desc=f"[{split_name}]", ncols=100):
            sample = split_data[i]
            stats["total"] += 1

            result = process_single_sample(
                sample=sample,
                index=global_index,
                wav_dir=wav_dir,
                target_sr=config.target_sr,
                bit_depth=config.bit_depth,
                skip_existing=config.skip_existing,
            )

            if result["success"]:
                stats["success"] += 1
                all_texts.append(result["text"])
                all_wav_paths.append(result["wav_path"])

                wav_path = Path(result["wav_path"])
                if config.skip_existing and wav_path.exists():
                    pass
            else:
                stats["errors"] += 1
                error_type = result["error"] or "Unknown"
                stats["error_details"][error_type] = (
                    stats["error_details"].get(error_type, 0) + 1
                )
                if stats["errors"] <= 20:
                    logger.warning(
                        f"  Sample {global_index}: {error_type}"
                    )

            global_index += 1

    elapsed = time.time() - start_time

    # --- Lưu file raw_texts.txt ---
    raw_text_path = work_dir / config.raw_text_file
    logger.info(f"\nĐang lưu {len(all_texts):,} text records → {raw_text_path}")

    with open(raw_text_path, "w", encoding="utf-8") as f:
        for text in all_texts:
            clean_text = text.replace("\n", " ").replace("\r", "").strip()
            f.write(clean_text + "\n")

    # --- Lưu file wav_paths.txt (mapping index ↔ wav path) ---
    wav_paths_file = work_dir / "wav_paths.txt"
    with open(wav_paths_file, "w", encoding="utf-8") as f:
        for wp in all_wav_paths:
            f.write(wp + "\n")

    # --- Thống kê ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("  KẾT QUẢ TRÍCH XUẤT AUDIO")
    logger.info("=" * 60)
    logger.info(f"  Thời gian      : {elapsed:.1f}s ({elapsed / 60:.1f} phút)")
    logger.info(f"  Tổng samples   : {stats['total']:,}")
    logger.info(f"  Thành công     : {stats['success']:,}")
    logger.info(f"  Lỗi           : {stats['errors']:,}")
    logger.info(f"  Wav dir        : {wav_dir}")
    logger.info(f"  Raw text file  : {raw_text_path}")
    logger.info(f"  Wav paths file : {wav_paths_file}")

    if stats["error_details"]:
        logger.info("")
        logger.info("  Chi tiết lỗi:")
        for err_type, count in sorted(
            stats["error_details"].items(), key=lambda x: -x[1]
        ):
            logger.info(f"    {err_type}: {count:,}")

    # --- Kiểm tra sơ bộ audio đầu ra ---
    sample_wavs = list(wav_dir.glob("*.wav"))[:3]
    if sample_wavs:
        logger.info("")
        logger.info("  Kiểm tra mẫu:")
        for wav_file in sample_wavs:
            info = sf.info(str(wav_file))
            logger.info(
                f"    {wav_file.name}: "
                f"sr={info.samplerate}, "
                f"channels={info.channels}, "
                f"duration={info.duration:.2f}s, "
                f"format={info.subtype}"
            )

    logger.info("=" * 60)

# MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 2: Giải mã audio từ dataset ViVoice → file .wav chuẩn"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Đường dẫn tới file config.yaml",
    )
    args = parser.parse_args()

    # --- Load .env ---
    env_path = Path(args.config).parent.parent / ".env"
    if not env_path.exists():
        env_path = Path(args.config).parent.parent.parent.parent / ".env"
    if env_path.exists():
        load_dotenv(str(env_path))

    # --- Load config ---
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
        sys.exit(1)

    config = ExtractConfig.from_yaml(str(config_path))

    # HF token từ .env
    hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if hf_token:
        config.hf_token = hf_token

    # --- Setup logging ---
    log_dir = Path(config.work_dir) / "logs"
    logger = setup_logging(log_dir)

    # --- Header ---
    logger.info("=" * 60)
    logger.info("  BƯỚC 2: GIẢI MÃ AUDIO → FILE .WAV CHUẨN")
    logger.info("=" * 60)
    logger.info(f"Config         : {config_path.resolve()}")
    logger.info(f"Target SR      : {config.target_sr} Hz")
    logger.info(f"Channels       : {config.channels} (Mono)")
    logger.info(f"Bit depth      : {config.bit_depth}-bit PCM")
    logger.info(f"Wav output dir : {Path(config.output_dir) / config.wav_subdir}")
    logger.info(f"Skip existing  : {config.skip_existing}")

    # --- Chạy ---
    try:
        extract_dataset(config, logger)
        logger.info("")
        logger.info("Bước tiếp theo: Chạy step3_phonemize.py để chuyển text → phoneme")
        logger.info("=" * 60)
    except Exception as e:
        logger.exception(f"THẤT BẠI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()