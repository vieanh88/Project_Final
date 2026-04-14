"""
=============================================================
  RUN PIPELINE — Chạy toàn bộ quy trình xử lý audio
=============================================================
Cách sử dụng:

  # Xử lý tất cả file trong thư mục INPUT_DIR
  python run_pipeline.py

  # Chỉ xử lý 1 file cụ thể (để test trước)
  python run_pipeline.py --file raw_audio/ten_file.mp3

  # Chỉ chạy từ bước N (skip các bước trước nếu đã có cache)
  python run_pipeline.py --start-step 2

  # Xem thống kê kết quả mà không chạy lại
  python run_pipeline.py --stats-only

Các bước:
  1 → Vocal Separation (Demucs)
  2 → Speaker Diarization (pyannote)
  3 → Speaker Verification (embedding)
  4 → Quality Filtering
  5 → Export & Normalize
=============================================================
"""

import os
import gc
import sys
import json
import logging
import argparse
import time
from pathlib import Path
from typing import List, Optional

import torch

# Import config và modules
import config as cfg
from pipeline_modules import (
    VocalSeparator, SpeakerDiarizer, SpeakerVerifier,
    QualityFilter, AudioNormalizer,
    Segment, ProcessingResult
)

# ============================================================
#  SETUP LOGGING
# ============================================================

def setup_logging():
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(cfg.LOG_FILE, encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=handlers
    )
    # Giảm noise từ các thư viện
    for noisy_lib in ["numba", "urllib3", "filelock", "huggingface_hub"]:
        logging.getLogger(noisy_lib).setLevel(logging.WARNING)

    return logging.getLogger("PIPELINE")

# ============================================================
#  HELPER FUNCTIONS
# ============================================================

def get_audio_files(input_dir: str) -> List[Path]:
    """Lấy danh sách tất cả file audio trong thư mục."""
    supported_ext = {".mp3", ".wav", ".flac", ".m4a", ".ogg", ".aac"}
    input_path = Path(input_dir)
    files = [
        f for f in sorted(input_path.iterdir())
        if f.suffix.lower() in supported_ext
    ]
    return files


def print_banner(logger, message: str):
    border = "=" * 60
    logger.info(f"\n{border}\n  {message}\n{border}")


def print_step(logger, step: int, name: str):
    logger.info(f"\n{'─'*50}\n  BƯỚC {step}: {name}\n{'─'*50}")


def save_segments_cache(segments: List[Segment], cache_path: str):
    """Lưu segments ra file JSON để resume nếu bị ngắt giữa chừng."""
    data = [
        {
            "speaker_id": s.speaker_id,
            "start": s.start,
            "end": s.end,
            "audio_path": s.audio_path,
            "is_ngan": s.is_ngan,
            "similarity_score": s.similarity_score,
            "snr_db": s.snr_db,
            "passed_quality": s.passed_quality,
            "output_path": s.output_path,
        }
        for s in segments
    ]
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_segments_cache(cache_path: str) -> Optional[List[Segment]]:
    """Load segments từ cache JSON nếu có."""
    cache_path = Path(cache_path)
    if not cache_path.exists():
        return None
    with open(cache_path, encoding="utf-8") as f:
        data = json.load(f)
    return [Segment(**d) for d in data]


def print_summary(result: ProcessingResult, logger):
    """In tóm tắt kết quả xử lý 1 file."""
    logger.info(
        f"\n📊 KẾT QUẢ: {Path(result.input_file).name}\n"
        f"   Tổng segments       : {result.total_segments}\n"
        f"   Giọng Ngạn (verify) : {result.ngan_segments}\n"
        f"   Pass quality filter : {result.passed_segments}\n"
        f"   Tỉ lệ thu được      : "
        f"{100*result.passed_segments/max(result.total_segments,1):.1f}%\n"
        f"   Lý do loại          : {result.failed_reason_counts}"
    )

# ============================================================
#  CORE PIPELINE — xử lý 1 file audio
# ============================================================

def process_single_file(
    audio_file: Path,
    logger: logging.Logger,
    start_step: int = 1,
) -> ProcessingResult:
    """
    Chạy toàn bộ pipeline cho 1 file audio.
    """
    result = ProcessingResult(input_file=str(audio_file))
    stem = audio_file.stem

    # Thư mục làm việc riêng cho file này
    file_workdir = Path(cfg.WORK_DIR) / stem
    file_workdir.mkdir(parents=True, exist_ok=True)

    cache_path = str(file_workdir / "segments_cache.json")

    # ──────────────────────────────────────────────────────
    #  BƯỚC 1: VOCAL SEPARATION
    # ──────────────────────────────────────────────────────
    if start_step <= 1:
        print_step(logger, 1, "Vocal Separation (Demucs)")
        t0 = time.time()

        separator = VocalSeparator(
            model=cfg.DEMUCS_MODEL,
            cpu_only=cfg.DEMUCS_CPU_ONLY
        )
        try:
            vocals_path = separator.separate(
                audio_path=str(audio_file),
                output_dir=str(file_workdir / "demucs_out")
            )
        finally:
            separator.release()

        result.vocals_file = vocals_path
        logger.info(f"Bước 1 xong ({time.time()-t0:.1f}s) → {vocals_path}")
    else:
        # Tìm file vocals.wav đã có
        vocals_candidates = list(Path(file_workdir / "demucs_out").rglob("vocals.wav"))
        if not vocals_candidates:
            raise FileNotFoundError(
                f"Không tìm thấy vocals.wav trong {file_workdir}/demucs_out. "
                f"Hãy chạy từ bước 1."
            )
        vocals_path = str(vocals_candidates[0])
        result.vocals_file = vocals_path
        logger.info(f"Bỏ qua bước 1 (cache) → {vocals_path}")

    # ──────────────────────────────────────────────────────
    #  BƯỚC 2: DIARIZATION
    # ──────────────────────────────────────────────────────
    if start_step <= 2:
        print_step(logger, 2, "Speaker Diarization (pyannote)")
        t0 = time.time()

        rttm_path = str(file_workdir / f"{stem}.rttm") if cfg.SAVE_RTTM else None
        diarizer = SpeakerDiarizer(
            hf_token=cfg.HF_TOKEN,
            model_name=cfg.PYANNOTE_MODEL,
            max_speakers=cfg.DIARIZATION_MAX_SPEAKERS,
            merge_gap=cfg.MERGE_GAP_SECONDS
        )
        try:
            segments = diarizer.diarize(
                vocals_path=vocals_path,
                rttm_output_path=rttm_path
            )
            dominant_speaker = diarizer.get_dominant_speaker(segments)
        finally:
            diarizer.release()

        result.total_segments = len(segments)
        save_segments_cache(segments, cache_path)

        # Lưu dominant speaker ID ra file
        dominant_path = file_workdir / "dominant_speaker.txt"
        dominant_path.write_text(dominant_speaker)

        logger.info(
            f"Bước 2 xong ({time.time()-t0:.1f}s) — "
            f"{len(segments)} segments, dominant: {dominant_speaker}"
        )
    else:
        segments = load_segments_cache(cache_path)
        dominant_path = file_workdir / "dominant_speaker.txt"
        if segments is None or not dominant_path.exists():
            raise FileNotFoundError("Cache bước 2 không tìm thấy. Chạy từ --start-step 2.")
        dominant_speaker = dominant_path.read_text().strip()
        result.total_segments = len(segments)
        logger.info(f"Bỏ qua bước 2 (cache) — {len(segments)} segments, dominant: {dominant_speaker}")

    # ──────────────────────────────────────────────────────
    #  BƯỚC 3: SPEAKER VERIFICATION
    # ──────────────────────────────────────────────────────
    if start_step <= 3:
        print_step(logger, 3, "Speaker Verification (ECAPA-TDNN embedding)")
        t0 = time.time()

        verifier = SpeakerVerifier(
            hf_token=cfg.HF_TOKEN,
            similarity_threshold=cfg.SPEAKER_SIMILARITY_THRESHOLD,
            n_reference=cfg.N_REFERENCE_SEGMENTS,
            min_ref_duration=cfg.MIN_REFERENCE_DURATION
        )
        try:
            # Bootstrap reference từ dominant speaker
            verifier.build_reference_from_dominant(segments, dominant_speaker)
            # Verify toàn bộ
            segments = verifier.verify_segments(segments)
        finally:
            verifier.release()

        save_segments_cache(segments, cache_path)
        result.ngan_segments = sum(1 for s in segments if s.is_ngan)
        logger.info(
            f"Bước 3 xong ({time.time()-t0:.1f}s) — "
            f"{result.ngan_segments}/{len(segments)} segments xác nhận giọng Ngạn"
        )
    else:
        segments = load_segments_cache(cache_path)
        result.ngan_segments = sum(1 for s in segments if s.is_ngan)
        logger.info(f"Bỏ qua bước 3 (cache) — {result.ngan_segments} segments giọng Ngạn")

    # ──────────────────────────────────────────────────────
    #  BƯỚC 4: QUALITY FILTER
    # ──────────────────────────────────────────────────────
    if start_step <= 4:
        print_step(logger, 4, "Quality Filtering")
        t0 = time.time()

        qfilter = QualityFilter(
            min_duration=cfg.MIN_SEGMENT_DURATION,
            max_duration=cfg.MAX_SEGMENT_DURATION,
            min_snr_db=cfg.MIN_SNR_DB,
            scream_ratio=cfg.SCREAM_RATIO_THRESHOLD,
            max_clipping_pct=cfg.MAX_CLIPPING_PERCENT,
            max_silence_ratio=cfg.MAX_SILENCE_RATIO
        )
        passed_segments, failed_reasons = qfilter.filter(segments)
        result.failed_reason_counts = failed_reasons
        result.passed_segments = len(passed_segments)

        save_segments_cache(segments, cache_path)
        logger.info(f"Bước 4 xong ({time.time()-t0:.1f}s) — {len(passed_segments)} segments pass")
    else:
        segments = load_segments_cache(cache_path)
        passed_segments = [s for s in segments if s.passed_quality]
        result.passed_segments = len(passed_segments)
        logger.info(f"Bỏ qua bước 4 (cache) — {len(passed_segments)} segments pass")

    # ──────────────────────────────────────────────────────
    #  BƯỚC 5: EXPORT & NORMALIZE
    # ──────────────────────────────────────────────────────
    if start_step <= 5:
        print_step(logger, 5, f"Export & Normalize → {cfg.OUTPUT_DIR}")
        t0 = time.time()

        normalizer = AudioNormalizer(
            output_dir=cfg.OUTPUT_DIR,
            sample_rate=cfg.OUTPUT_SAMPLE_RATE,
            target_lufs=cfg.TARGET_LUFS,
            silence_pad_ms=cfg.SILENCE_PAD_MS
        )
        normalizer.export_segments(passed_segments, source_name=stem)
        logger.info(f"Bước 5 xong ({time.time()-t0:.1f}s)")

    result.segments = segments
    return result

# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline xử lý audio giọng Nguyễn Ngọc Ngạn cho XTTS v2"
    )
    parser.add_argument(
        "--file", type=str, default=None,
        help="Chỉ xử lý 1 file cụ thể (để test). Mặc định: xử lý tất cả file trong INPUT_DIR"
    )
    parser.add_argument(
        "--start-step", type=int, default=1, choices=[1, 2, 3, 4, 5],
        help="Bắt đầu từ bước nào (mặc định: 1). Hữu ích khi resume sau lỗi."
    )
    parser.add_argument(
        "--stats-only", action="store_true",
        help="Chỉ in thống kê từ cache, không chạy lại pipeline"
    )
    args = parser.parse_args()

    # Setup
    setup_logging()
    logger = logging.getLogger("PIPELINE")

    # Tạo thư mục
    for d in [cfg.INPUT_DIR, cfg.OUTPUT_DIR, cfg.WORK_DIR]:
        Path(d).mkdir(parents=True, exist_ok=True)

    # Kiểm tra HF token
    if cfg.HF_TOKEN == "hf_YOUR_TOKEN_HERE":
        logger.error(
            "Chưa cấu hình HF_TOKEN!\n"
            "Cách 1: Set biến môi trường → export HF_TOKEN=hf_xxxx\n"
            "Cách 2: Sửa trực tiếp trong config.py"
        )
        sys.exit(1)

    print_banner(logger, "PIPELINE XỬ LÝ GIỌNG NGUYỄN NGỌC NGẠN")
    logger.info(f"Device: {'CUDA (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'CPU'}")
    logger.info(f"Input: {cfg.INPUT_DIR}")
    logger.info(f"Output: {cfg.OUTPUT_DIR}")

    # Lấy danh sách file cần xử lý
    if args.file:
        audio_files = [Path(args.file)]
        if not audio_files[0].exists():
            logger.error(f"File không tồn tại: {args.file}")
            sys.exit(1)
    else:
        audio_files = get_audio_files(cfg.INPUT_DIR)
        if not audio_files:
            logger.error(f"Không tìm thấy file audio trong {cfg.INPUT_DIR}")
            sys.exit(1)

    logger.info(f"Sẽ xử lý {len(audio_files)} file audio.")

    # ── STATS ONLY MODE ──
    if args.stats_only:
        logger.info("Mode: stats-only — đọc cache và in thống kê.")
        for af in audio_files:
            stem = af.stem
            cache_path = Path(cfg.WORK_DIR) / stem / "segments_cache.json"
            segments = load_segments_cache(str(cache_path))
            if segments:
                ngan = [s for s in segments if s.is_ngan]
                passed = [s for s in segments if s.passed_quality]
                total_dur = sum(s.duration for s in passed)
                logger.info(
                    f"{af.name}: total={len(segments)} | "
                    f"ngan={len(ngan)} | passed={len(passed)} | "
                    f"duration={total_dur/60:.1f}min"
                )
            else:
                logger.info(f"{af.name}: chưa có cache")
        return

    # ── PROCESSING LOOP ──
    all_results = []
    total_start = time.time()

    for i, audio_file in enumerate(audio_files, 1):
        print_banner(
            logger,
            f"FILE {i}/{len(audio_files)}: {audio_file.name}"
        )
        file_start = time.time()

        try:
            result = process_single_file(
                audio_file=audio_file,
                logger=logger,
                start_step=args.start_step,
            )
            all_results.append(result)
            print_summary(result, logger)
            logger.info(f"⏱  Thời gian xử lý: {(time.time()-file_start)/60:.1f} phút")

        except KeyboardInterrupt:
            logger.warning("Bị ngắt bởi người dùng. Thoát.")
            break
        except Exception as e:
            logger.error(f"Lỗi khi xử lý {audio_file.name}: {e}", exc_info=True)
            logger.info("Bỏ qua file này, tiếp tục...")
            continue
        finally:
            # Đảm bảo VRAM được giải phóng giữa các file
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

    # ── TỔNG KẾT ──
    if all_results:
        print_banner(logger, "TỔNG KẾT TOÀN BỘ")
        total_passed = sum(r.passed_segments for r in all_results)
        total_time = (time.time() - total_start) / 60

        logger.info(f"Tổng files xử lý      : {len(all_results)}/{len(audio_files)}")
        logger.info(f"Tổng segments thu được : {total_passed}")
        logger.info(f"Tổng thời gian chạy   : {total_time:.1f} phút")

        # Lưu metadata tổng hợp
        all_segments = [s for r in all_results for s in r.segments]
        metadata_path = Path(cfg.OUTPUT_DIR) / "metadata.csv"
        normalizer = AudioNormalizer(output_dir=cfg.OUTPUT_DIR)
        normalizer.save_metadata_csv(all_segments, str(metadata_path))

        logger.info(f"\n✅ Hoàn tất! Dataset lưu tại: {cfg.OUTPUT_DIR}")
        logger.info(f"   File metadata: {metadata_path}")

        # Tính tổng thời lượng dataset
        passed_segs = [s for s in all_segments if s.passed_quality and s.output_path]
        total_duration_min = sum(s.duration for s in passed_segs) / 60
        logger.info(f"   Tổng thời lượng dataset: {total_duration_min:.1f} phút")

        # Nhắc nhở về XTTS v2 requirements
        if total_duration_min < 30:
            logger.warning(
                f"⚠️  Dataset chỉ có {total_duration_min:.1f} phút. "
                f"XTTS v2 fine-tuning cần tối thiểu 30-60 phút để đạt chất lượng tốt. "
                f"Hãy thêm nhiều audio nguồn hơn."
            )
        else:
            logger.info(
                f"✅ {total_duration_min:.1f} phút — Đủ điều kiện cho XTTS v2 fine-tuning!"
            )

if __name__ == "__main__":
    main()