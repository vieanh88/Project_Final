"""
=============================================================
  BƯỚC 2: AUDIO RESTORATION — DeepFilterNet3
=============================================================
Mục tiêu: Loại bỏ noise còn sót sau Demucs (reverb, white noise,
          music artifacts) để thu được giọng nói "khô" (dry voice).

Input  : workdir/step01_vocals/  (*_vocals.wav)
Output : workdir/step02_restored/  (*_restored.wav)

Cách chạy:
  python step02_audio_restoration.py
  python step02_audio_restoration.py --input workdir/step01_vocals/ten_file_vocals.wav
  python step02_audio_restoration.py --dry-run

Lưu ý:
  - DeepFilterNet3 chạy tốt trên cả CPU và GPU
  - VRAM cần: ~500MB (rất nhẹ, không ảnh hưởng các bước khác)
  - Nếu lỗi cài deepfilternet:
      pip install deepfilternet --extra-index-url https://download.pytorch.org/whl/cu121
=============================================================
"""
import os
import sys
import time
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import yaml

#  CONFIG LOADER
def load_config(config_path: str = "config.yaml") -> dict:
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"[LỖI] Không tìm thấy file config: {config_path}")
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
    # Tắt warning không cần thiết từ DeepFilterNet
    for noisy in ["df", "df.enhance", "df.model", "torch"]:
        logging.getLogger(noisy).setLevel(logging.WARNING)
    return logging.getLogger("step02")

#  HELPERS
def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"

def get_input_files(step01_dir: str, single_file: Optional[str] = None) -> List[Path]:
    """Lấy danh sách *_vocals.wav từ output bước 1."""
    if single_file:
        p = Path(single_file)
        if not p.exists():
            print(f"[LỖI] File không tồn tại: {single_file}")
            sys.exit(1)
        return [p]

    step01_path = Path(step01_dir)
    if not step01_path.exists():
        print(f"[LỖI] Thư mục output bước 1 không tồn tại: {step01_dir}")
        print("      Hãy chạy step01_vocal_isolation.py trước.")
        sys.exit(1)

    files = sorted(step01_path.glob("*_vocals.wav"))
    if not files:
        print(f"[LỖI] Không tìm thấy file *_vocals.wav trong: {step01_dir}")
        print("      Hãy chạy step01_vocal_isolation.py trước.")
        sys.exit(1)
    return files

def get_output_name(input_path: Path) -> str:
    """
    Chuyển tên: ten_file_vocals.wav → ten_file_restored.wav
    """
    stem = input_path.stem  # "ten_file_vocals"
    if stem.endswith("_vocals"):
        base = stem[: -len("_vocals")]
    else:
        base = stem
    return f"{base}_restored.wav"

#  DEEPFILTER ENGINE
class DeepFilterEngine:
    """
    Wrapper cho DeepFilterNet.
    Lazy-load model để tiết kiệm thời gian khởi động khi dry-run.
    """

    def __init__(self, model_version: str, attenuation_limit: float, use_gpu: bool):
        self.model_version = model_version
        self.attenuation_limit = attenuation_limit
        self.use_gpu = use_gpu
        self._model = None
        self._df_state = None
        self._model_sr = None  # Sample rate mà model expect

    def _load(self, logger: logging.Logger):
        """Load DeepFilterNet model (chỉ load 1 lần)."""
        if self._model is not None:
            return

        logger.info(f"  Load {self.model_version} model ...")
        try:
            from df.enhance import init_df
        except ImportError:
            logger.error("DeepFilterNet chưa được cài đặt!")
            logger.error("Cài đặt: pip install deepfilternet")
            logger.error("Hoặc: pip install deepfilternet --extra-index-url "
                         "https://download.pytorch.org/whl/cu121")
            sys.exit(1)

        try:
            # init_df trả về (model, df_state, suffix)
            # df_state chứa sample_rate mà model expect
            result = init_df(
                model_base_dir=None,  # Dùng model mặc định (tự download)
                config_allow_defaults=True,
                log_level="WARNING",
            )
            # API có thể trả về 2 hoặc 3 giá trị tùy version
            if len(result) == 3:
                self._model, self._df_state, _ = result
            else:
                self._model, self._df_state = result

            self._model_sr = self._df_state.sr()
            logger.info(
                f"  Model loaded — sample rate: {self._model_sr} Hz, "
                f"attenuation limit: {self.attenuation_limit}"
            )
        except Exception as e:
            logger.error(f"  Lỗi khi load DeepFilterNet: {e}")
            raise

    def enhance(
        self,
        audio: np.ndarray,
        sample_rate: int,
        logger: logging.Logger,
    ) -> Tuple[np.ndarray, int]:
        """
        Chạy DeepFilterNet trên 1 mảng audio.

        Args:
            audio:       numpy array, shape (samples,) hoặc (channels, samples)
            sample_rate: sample rate của audio input
            logger:      logger instance

        Returns:
            (enhanced_audio, output_sample_rate)
        """
        import torch
        from df.enhance import enhance as df_enhance

        self._load(logger)

        # Chuẩn hóa sang float32
        audio = audio.astype(np.float32)

        # Đảm bảo audio là 2D: (channels, samples) theo chuẩn DeepFilterNet
        if audio.ndim == 1:
            audio = audio[np.newaxis, :]  # (1, samples)
        elif audio.ndim == 2 and audio.shape[0] > audio.shape[1]:
            # Trường hợp shape là (samples, channels) → transpose
            audio = audio.T

        # Resample sang sample rate của model nếu khác
        if sample_rate != self._model_sr:
            import torchaudio
            audio_tensor = torch.from_numpy(audio)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sample_rate,
                new_freq=self._model_sr
            )
            audio_tensor = resampler(audio_tensor)
            audio = audio_tensor.numpy()

        # Chuyển sang tensor
        audio_tensor = torch.from_numpy(audio)

        # Chuyển sang tensor (giữ trên CPU — DeepFilterNet tự quản lý device nội bộ)
        # KHÔNG dùng .cuda() ở đây: df.analysis() bên trong DeepFilterNet gọi
        # .numpy() trực tiếp trên tensor đầu vào, nên tensor PHẢI ở CPU.
        # Model đã được load đúng device bởi init_df; df_enhance() tự lo phần còn lại
        # Không đưa lên GPU
        #if self.use_gpu and torch.cuda.is_available():
        #    audio_tensor = audio_tensor.cuda()

        # Chạy DeepFilterNet enhance
        enhanced_tensor = df_enhance(
            model=self._model,
            df_state=self._df_state,
            audio=audio_tensor,
            atten_lim_db=self._db_from_ratio(self.attenuation_limit),
            pad=True,
        )

        # Chuyển về numpy, đảm bảo về CPU
        enhanced = enhanced_tensor.cpu().numpy()

        # Flatten về mono nếu cần (lấy channel đầu tiên)
        if enhanced.ndim == 2:
            enhanced = enhanced[0]

        return enhanced, self._model_sr

    def _db_from_ratio(self, ratio: float) -> float:
        """
        Chuyển attenuation_limit từ ratio [0-1] sang dB.
        ratio = 0.97 → ~-15dB attenuation limit
        Công thức: dB = 20 * log10(1 - ratio) nhưng DeepFilterNet
        dùng tham số trực tiếp là dB, nên ta map:
        0.0 → 0 dB (không lọc), 1.0 → 100 dB (lọc tối đa)
        """
        # Map tuyến tính: ratio 0.97 → 100 * 0.97 = 97 dB
        # Đây là cách đơn giản và dễ hiểu nhất
        return ratio * 100.0

    def release(self):
        """Giải phóng bộ nhớ."""
        self._model = None
        self._df_state = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

#  PROCESS 1 FILE
def process_file(
    input_path: Path,
    output_path: Path,
    engine: DeepFilterEngine,
    logger: logging.Logger,
) -> bool:
    """
    Chạy DeepFilterNet trên 1 file WAV.

    Returns:
        True nếu thành công, False nếu thất bại.
    """
    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile chưa cài. Chạy: pip install soundfile")
        sys.exit(1)

    # Đọc audio
    try:
        audio, sr = sf.read(str(input_path), dtype="float32", always_2d=False)
    except Exception as e:
        logger.error(f"  Lỗi đọc file: {e}")
        return False

    duration_s = len(audio) / sr if audio.ndim == 1 else audio.shape[0] / sr
    logger.info(f"  Audio: {sr}Hz, {duration_s:.1f}s")

    # Xử lý từng chunk nếu file dài (tránh OOM với file rất dài)
    # DeepFilterNet xử lý tốt file dài, nhưng chia nhỏ giúp ổn định hơn
    CHUNK_DURATION_S = 60  # Xử lý từng đoạn 60 giây
    chunk_samples = int(CHUNK_DURATION_S * sr)

    if audio.ndim > 1:
        total_samples = audio.shape[0]
    else:
        total_samples = len(audio)

    if total_samples <= chunk_samples:
        # File ngắn → xử lý 1 lần
        enhanced, out_sr = engine.enhance(audio, sr, logger)
    else:
        # File dài → chia chunk
        n_chunks = (total_samples + chunk_samples - 1) // chunk_samples
        logger.info(f"  File dài — chia {n_chunks} chunks ({CHUNK_DURATION_S}s/chunk)")

        enhanced_chunks = []
        if audio.ndim > 1:
            # Multichannel
            for i in range(n_chunks):
                chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]
                enh_chunk, out_sr = engine.enhance(chunk, sr, logger)
                enhanced_chunks.append(enh_chunk)
        else:
            # Mono
            for i in range(n_chunks):
                chunk = audio[i * chunk_samples: (i + 1) * chunk_samples]
                enh_chunk, out_sr = engine.enhance(chunk, sr, logger)
                enhanced_chunks.append(enh_chunk)

        enhanced = np.concatenate(enhanced_chunks, axis=-1)

    # Normalize nhẹ để tránh clipping (không thay đổi âm lượng đáng kể)
    peak = np.max(np.abs(enhanced))
    if peak > 0.99:
        enhanced = enhanced * (0.99 / peak)

    # Lưu output
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(
            str(output_path),
            enhanced,
            out_sr,
            subtype="PCM_16",
            format="WAV",
        )
    except Exception as e:
        logger.error(f"  Lỗi khi lưu file: {e}")
        return False

    out_size_mb = output_path.stat().st_size / 1024**2
    logger.info(f"  Restored SR: {out_sr} Hz | Size: {out_size_mb:.1f} MB")
    return True

#  MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 2: Khử noise bằng DeepFilterNet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--input", default=None,
        help="Xử lý 1 file cụ thể (phải là *_vocals.wav từ bước 1)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Liệt kê file sẽ xử lý mà không chạy"
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg = cfg["step02"]

    # Setup paths
    work_dir = Path(paths_cfg["work_dir"])
    step01_dir = work_dir / "step01_vocals"
    step_output_dir = work_dir / "step02_restored"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step02.log")

    logger = setup_logging(log_path)

    # Header
    logger.info("=" * 60)
    logger.info("  BƯỚC 2: AUDIO RESTORATION (DeepFilterNet)")
    logger.info("=" * 60)
    logger.info(f"  Model    : {step_cfg['model_version']}")
    logger.info(f"  Atten.   : {step_cfg['attenuation_limit']} (lọc noise ~{step_cfg['attenuation_limit']*100:.0f}dB)")
    logger.info(f"  GPU      : {step_cfg['use_gpu']}")
    logger.info(f"  Input    : {step01_dir}")
    logger.info(f"  Output   : {step_output_dir}")

    # Lấy danh sách file
    input_files = get_input_files(str(step01_dir), args.input)
    logger.info(f"  Files    : {len(input_files)} file cần xử lý")

    # Dry run
    if args.dry_run:
        logger.info("\n[DRY RUN] Danh sách file:")
        for i, f in enumerate(input_files, 1):
            out = step_output_dir / get_output_name(f)
            status = "✓ đã có" if out.exists() else "→ cần xử lý"
            size_mb = f.stat().st_size / 1024**2
            logger.info(f"  {i:3d}. {f.name} ({size_mb:.1f}MB) → {out.name} [{status}]")
        sys.exit(0)

    # Khởi tạo engine (chưa load model)
    engine = DeepFilterEngine(
        model_version=step_cfg["model_version"],
        attenuation_limit=step_cfg["attenuation_limit"],
        use_gpu=step_cfg["use_gpu"],
    )

    results = {"success": [], "skipped": [], "failed": []}
    total_start = time.time()

    try:
        for idx, input_file in enumerate(input_files, 1):
            out_name = get_output_name(input_file)
            output_file = step_output_dir / out_name

            logger.info(f"\n[{idx}/{len(input_files)}] {input_file.name}")

            # Kiểm tra cache
            if step_cfg["skip_existing"] and output_file.exists():
                size_mb = output_file.stat().st_size / 1024**2
                logger.info(f"  Skip (cache): {out_name} ({size_mb:.1f} MB)")
                results["skipped"].append(input_file.name)
                continue

            t_start = time.time()
            success = process_file(input_file, output_file, engine, logger)
            elapsed = time.time() - t_start

            if success:
                logger.info(f"  ✓ Xong ({format_duration(elapsed)}) → {out_name}")
                results["success"].append(input_file.name)
            else:
                logger.error(f"  ✗ Thất bại: {input_file.name}")
                results["failed"].append(input_file.name)
                # Xóa file output dở dang nếu có
                if output_file.exists():
                    output_file.unlink()

    finally:
        engine.release()

    # Tổng kết
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 2")
    logger.info("=" * 60)
    logger.info(f"  ✓ Thành công : {len(results['success'])} file")
    logger.info(f"  ⏭ Bỏ qua    : {len(results['skipped'])} file (cache)")
    logger.info(f"  ✗ Thất bại  : {len(results['failed'])} file")
    logger.info(f"  Tổng thời gian: {format_duration(total_elapsed)}")

    if results["failed"]:
        logger.warning("\nFile thất bại:")
        for f in results["failed"]:
            logger.warning(f"  - {f}")
        logger.warning("Gợi ý: Kiểm tra file có bị corrupt không bằng: ffprobe <file>")

    # Liệt kê output
    output_files = sorted(step_output_dir.glob("*_restored.wav"))
    logger.info(f"\n  Output ({len(output_files)} file):")
    for f in output_files:
        logger.info(f"    {f.name} ({f.stat().st_size / 1024**2:.1f} MB)")

    if results["failed"]:
        sys.exit(1)

    logger.info("\n→ Chạy tiếp: python step03_vad_slicing.py")

if __name__ == "__main__":
    main()