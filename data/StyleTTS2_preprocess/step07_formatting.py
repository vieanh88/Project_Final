"""
=============================================================
  BƯỚC 7: FORMATTING — Chuẩn hóa cho StyleTTS2-lite-vi
=============================================================
Mục tiêu: Chuẩn hóa toàn bộ dataset về đúng format StyleTTS2-lite-vi:
  - Sample rate : 24000 Hz
  - Channel     : Mono
  - Bit depth   : 16-bit PCM WAV
  - Loudness    : Normalized về -20 LUFS (EBU R128)
  - Padding     : 50ms silence đầu/cuối mỗi file
  - Filelist    : UTF-8, format "path|text"

Input  : workdir/step06_quality/  (quality_manifest.csv + *.wav + *.txt)
Output : output_dataset/
            wavs/
                ngan_0001.wav
                ngan_0002.wav
                ...
            filelist_train.txt   ← 95% data (LJSpeech format)
            filelist_val.txt     ← 5%  data (để validate khi training)
            filelist_all.txt     ← toàn bộ
            dataset_info.json    ← metadata tóm tắt dataset

Cách chạy:
  python step07_formatting.py
  python step07_formatting.py --dry-run
  python step07_formatting.py --val-ratio 0.1   # 10% validation

Lưu ý:
  - Bước này là bước cuối, output_dataset/ là thư mục nộp cho StyleTTS2
  - Tên file được đánh số lại từ đầu (ngan_0001.wav...) để đồng nhất
  - File filelist_train.txt đọc trực tiếp bằng StyleTTS2 training script
=============================================================
"""

import os
import sys
import csv
import json
import time
import random
import logging
import argparse
from pathlib import Path
from typing import List, Optional, Dict, Tuple, NamedTuple

import numpy as np
import yaml


# ============================================================
#  CONFIG & LOGGING
# ============================================================

def load_config(config_path: str = "config.yaml") -> dict:
    p = Path(config_path)
    if not p.exists():
        print(f"[LỖI] Không tìm thấy config: {config_path}")
        sys.exit(1)
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="a"),
        ],
    )
    for lib in ["numba", "librosa", "soundfile"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("step07")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m = int(seconds // 60)
    s = seconds % 60
    if m < 60:
        return f"{m}m {s:.0f}s"
    return f"{m // 60}h {m % 60}m"


# ============================================================
#  DATA STRUCTURES
# ============================================================

class FormattedRecord(NamedTuple):
    original_wav:  Path
    output_wav:    Path
    transcript:    str
    duration_in:   float   # duration trước khi format
    duration_out:  float   # duration sau khi format (có padding)
    source_file:   str


# ============================================================
#  MANIFEST LOADER
# ============================================================

def load_quality_manifest(
    manifest_path: Path,
    logger: logging.Logger,
) -> List[Dict]:
    """Đọc quality_manifest.csv từ bước 6."""
    if not manifest_path.exists():
        logger.error(f"Không tìm thấy manifest: {manifest_path}")
        logger.error("Hãy chạy step06_quality_check.py trước.")
        sys.exit(1)

    rows = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav = Path(row["wav_path"])
            if not wav.exists():
                logger.warning(f"  WAV không tồn tại, bỏ qua: {wav.name}")
                continue
            rows.append({
                "source_file": row.get("source_file", "unknown"),
                "slice_index": int(row.get("slice_index", 0)),
                "duration":    float(row.get("duration_s", 0)),
                "wav_path":    wav,
                "txt_path":    Path(row["txt_path"]) if row.get("txt_path") else None,
                "transcript":  row.get("transcript", ""),
                "mos_ovrl":    float(row.get("mos_ovrl", 0)),
            })

    logger.info(f"  Quality manifest: {len(rows)} records")
    return rows


# ============================================================
#  AUDIO PROCESSOR
# ============================================================

class AudioProcessor:
    """
    Xử lý audio: resample, convert mono, normalize loudness, pad silence.
    """

    def __init__(
        self,
        target_sr:     int   = 24000,
        target_ch:     int   = 1,
        target_lufs:   float = -20.0,
        silence_pad_ms: int  = 50,
    ):
        self.target_sr      = target_sr
        self.target_ch      = target_ch
        self.target_lufs    = target_lufs
        self.silence_pad_s  = silence_pad_ms / 1000.0
        self._check_deps()

    def _check_deps(self):
        """Kiểm tra dependencies bắt buộc."""
        missing = []
        try:
            import soundfile
        except ImportError:
            missing.append("soundfile")
        try:
            import librosa
        except ImportError:
            missing.append("librosa")
        if missing:
            print(f"[LỖI] Thiếu thư viện: {', '.join(missing)}")
            print(f"      Cài đặt: pip install {' '.join(missing)}")
            sys.exit(1)

    def process(
        self,
        input_path:  Path,
        output_path: Path,
        logger:      logging.Logger,
    ) -> Tuple[bool, float]:
        """
        Xử lý 1 file: resample → mono → normalize → pad → save.

        Returns:
            (success, output_duration_seconds)
        """
        import librosa
        import soundfile as sf

        # ── Load ───────────────────────────────────────────────
        try:
            audio, sr = librosa.load(
                str(input_path),
                sr=self.target_sr,   # Resample ngay khi load
                mono=True,           # Convert sang mono
            )
        except Exception as e:
            logger.debug(f"  Lỗi load {input_path.name}: {e}")
            return False, 0.0

        if len(audio) == 0:
            logger.debug(f"  Audio rỗng: {input_path.name}")
            return False, 0.0

        # ── Normalize loudness ─────────────────────────────────
        audio = self._normalize_loudness(audio)

        # ── Trim silence thừa ở đầu/cuối ──────────────────────
        # Trim trước khi pad để tránh pad lên silence dư
        audio, _ = librosa.effects.trim(
            audio,
            top_db=40,       # Cắt vùng < -40dBFS so với peak
            frame_length=256,
            hop_length=64,
        )

        if len(audio) == 0:
            logger.debug(f"  Audio bị trim hết: {input_path.name}")
            return False, 0.0

        # ── Pad silence ────────────────────────────────────────
        pad_samples = int(self.silence_pad_s * self.target_sr)
        silence     = np.zeros(pad_samples, dtype=np.float32)
        audio       = np.concatenate([silence, audio, silence])

        # ── Final safety clip ──────────────────────────────────
        audio = np.clip(audio, -1.0, 1.0)

        # ── Save ───────────────────────────────────────────────
        output_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            sf.write(
                str(output_path),
                audio,
                self.target_sr,
                subtype="PCM_16",
                format="WAV",
            )
        except Exception as e:
            logger.debug(f"  Lỗi lưu {output_path.name}: {e}")
            return False, 0.0

        duration = len(audio) / self.target_sr
        return True, duration

    def _normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize âm lượng về target LUFS bằng RMS-based approach.

        Dùng pyloudnorm nếu có (chuẩn EBU R128 thực sự).
        Fallback về RMS normalization nếu không có pyloudnorm.
        """
        # ── Thử pyloudnorm (chuẩn nhất) ───────────────────────
        try:
            import pyloudnorm as pyln
            meter = pyln.Meter(self.target_sr)  # EBU R128
            loudness = meter.integrated_loudness(audio)
            if not np.isfinite(loudness) or loudness < -70:
                # Quá yên tĩnh → fallback RMS
                return self._rms_normalize(audio)
            normalized = pyln.normalize.loudness(audio, loudness, self.target_lufs)
            return np.clip(normalized.astype(np.float32), -1.0, 1.0)
        except ImportError:
            pass
        except Exception:
            pass

        # ── Fallback: RMS normalization ────────────────────────
        return self._rms_normalize(audio)

    def _rms_normalize(self, audio: np.ndarray) -> np.ndarray:
        """
        RMS normalization về target LUFS (approximate).
        -20 LUFS ≈ RMS = 0.1 = -20 dBFS
        """
        rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
        if rms < 1e-9:
            return audio  # Silence — không normalize

        target_rms = 10 ** (self.target_lufs / 20.0)
        gain = target_rms / rms

        # Safety: không boost quá mạnh (tránh clipping khi signal rất yếu)
        gain = min(gain, 10.0)

        return np.clip((audio * gain).astype(np.float32), -1.0, 1.0)


# ============================================================
#  FILELIST WRITER
# ============================================================

def write_filelists(
    records:    List[FormattedRecord],
    output_dir: Path,
    val_ratio:  float,
    encoding:   str,
    logger:     logging.Logger,
) -> Tuple[int, int]:
    """
    Tạo 3 filelist:
    - filelist_all.txt   — tất cả
    - filelist_train.txt — (1 - val_ratio) * total
    - filelist_val.txt   — val_ratio * total

    Shuffle trước khi split để validation set đa dạng.
    Returns: (n_train, n_val)
    """
    # Shuffle với seed cố định để reproducible
    shuffled = list(records)
    random.seed(42)
    random.shuffle(shuffled)

    n_val   = max(1, int(len(shuffled) * val_ratio))
    n_train = len(shuffled) - n_val

    val_records   = shuffled[:n_val]
    train_records = shuffled[n_val:]

    def write_list(path: Path, recs: List[FormattedRecord]):
        with open(path, "w", encoding=encoding) as f:
            for r in recs:
                # LJSpeech format: "absolute_or_relative_path|text"
                f.write(f"{r.output_wav}|{r.transcript}\n")

    write_list(output_dir / "filelist_all.txt",   shuffled)
    write_list(output_dir / "filelist_train.txt", train_records)
    write_list(output_dir / "filelist_val.txt",   val_records)

    logger.info(f"  filelist_train.txt: {n_train} records")
    logger.info(f"  filelist_val.txt  : {n_val}   records")
    logger.info(f"  filelist_all.txt  : {len(shuffled)} records")

    return n_train, n_val


# ============================================================
#  DATASET INFO JSON
# ============================================================

def save_dataset_info(
    records:    List[FormattedRecord],
    output_dir: Path,
    step_cfg:   dict,
    n_train:    int,
    n_val:      int,
    logger:     logging.Logger,
):
    """Lưu metadata tóm tắt dataset ra JSON."""
    durations = [r.duration_out for r in records]
    total_dur = sum(durations)

    # Phân phối thời lượng
    dur_bins = {"3-4s": 0, "4-5s": 0, "5-6s": 0, "6-7s": 0, "7-8s": 0, "8-9s": 0, "9-10s": 0, "10-15s": 0, "15s-18s": 0}
    for d in durations:
        for lo, hi, label in [
            (3, 4, "3-4s"), (4, 5, "4-5s"), (5, 6, "5-6s"),
            (6, 7, "6-7s"), (7, 8, "7-8s"), (8, 9, "8-9s"),
            (9, 10, "9-10s"), (10, 15, "10-15s"), (15, 18, "15s-18s")
        ]:
            if lo <= d < hi:
                dur_bins[label] += 1
                break

    # Nguồn audio
    source_counts: Dict[str, int] = {}
    for r in records:
        source_counts[r.source_file] = source_counts.get(r.source_file, 0) + 1

    info = {
        "dataset_info": {
            "created_at":    time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipeline":      "StyleTTS2-lite-vi Data Pipeline v2",
            "speaker":       "Nguyễn Ngọc Ngạn",
        },
        "audio_format": {
            "sample_rate":   step_cfg["sample_rate"],
            "channels":      step_cfg["channels"],
            "bit_depth":     step_cfg["bit_depth"],
            "format":        step_cfg["format"],
            "target_lufs":   step_cfg["target_lufs"],
        },
        "statistics": {
            "total_files":         len(records),
            "total_duration_s":    round(total_dur, 2),
            "total_duration_min":  round(total_dur / 60, 2),
            "avg_duration_s":      round(np.mean(durations), 2) if durations else 0,
            "min_duration_s":      round(min(durations), 2) if durations else 0,
            "max_duration_s":      round(max(durations), 2) if durations else 0,
            "duration_distribution": dur_bins,
        },
        "splits": {
            "train": n_train,
            "val":   n_val,
            "total": len(records),
        },
        "sources": source_counts,
        "styletts2_readiness": {
            "min_recommended_min": 30,
            "current_min":         round(total_dur / 60, 1),
            "ready":               total_dur >= 1800,
            "note": (
                "✅ Đủ điều kiện fine-tuning" if total_dur >= 1800
                else f"⚠ Cần thêm {(1800 - total_dur)/60:.1f} phút nữa"
            ),
        },
    }

    out_path = output_dir / "dataset_info.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    logger.info(f"  dataset_info.json: {out_path}")
    return info


# ============================================================
#  VERIFICATION
# ============================================================

def verify_output_sample(
    records: List[FormattedRecord],
    n_samples: int,
    logger: logging.Logger,
):
    """
    Verify ngẫu nhiên N file output để đảm bảo format đúng.
    In cảnh báo nếu file bị lỗi.
    """
    import soundfile as sf

    sample = random.sample(records, min(n_samples, len(records)))
    errors = 0

    logger.info(f"\n  Verify {len(sample)} file output ngẫu nhiên ...")
    for r in sample:
        try:
            info = sf.info(str(r.output_wav))
            ok   = (
                info.samplerate == 24000
                and info.channels == 1
                and "PCM_16" in info.subtype
            )
            if not ok:
                logger.warning(
                    f"  ⚠ Format sai: {r.output_wav.name} "
                    f"sr={info.samplerate} ch={info.channels} subtype={info.subtype}"
                )
                errors += 1
        except Exception as e:
            logger.warning(f"  ⚠ Không đọc được: {r.output_wav.name}: {e}")
            errors += 1

    if errors == 0:
        logger.info(f"  ✓ Tất cả {len(sample)} file verify OK (24kHz/mono/PCM_16)")
    else:
        logger.warning(f"  {errors}/{len(sample)} file có vấn đề về format!")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bước 7: Formatting & normalization cho StyleTTS2-lite-vi",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",    default="config.yaml")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="In thống kê, không xử lý file"
    )
    parser.add_argument(
        "--val-ratio", type=float, default=None,
        help="Tỉ lệ validation set (mặc định theo config, thường 0.05)"
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Bỏ qua file WAV đã có trong output (resume)"
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg  = cfg["step07"]

    work_dir        = Path(paths_cfg["work_dir"])
    step06_dir      = work_dir / "step06_quality"
    output_dir      = Path(paths_cfg["output_dir"])
    wavs_dir        = output_dir / "wavs"
    wavs_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step07.log")

    logger = setup_logging(log_path)

    val_ratio = args.val_ratio if args.val_ratio is not None else 0.05

    logger.info("=" * 60)
    logger.info("  BƯỚC 7: FORMATTING (StyleTTS2-lite-vi)")
    logger.info("=" * 60)
    logger.info(f"  Target SR    : {step_cfg['sample_rate']} Hz")
    logger.info(f"  Target ch    : {step_cfg['channels']} (mono)")
    logger.info(f"  Bit depth    : {step_cfg['bit_depth']}-bit")
    logger.info(f"  Target LUFS  : {step_cfg['target_lufs']} LUFS")
    logger.info(f"  Silence pad  : {step_cfg['silence_pad_ms']} ms")
    logger.info(f"  Val ratio    : {val_ratio:.0%}")
    logger.info(f"  Prefix       : {step_cfg['output_prefix']}")
    logger.info(f"  Input        : {step06_dir}")
    logger.info(f"  Output       : {output_dir}")

    # Đọc manifest bước 6
    manifest_path = step06_dir / "quality_manifest.csv"
    all_rows      = load_quality_manifest(manifest_path, logger)

    if not all_rows:
        logger.error("Manifest rỗng. Kiểm tra lại bước 6.")
        sys.exit(1)

    total_input_dur = sum(r["duration"] for r in all_rows)
    logger.info(
        f"  {len(all_rows)} slices, "
        f"tổng {format_duration(total_input_dur)}"
    )

    # Dry run
    if args.dry_run:
        logger.info("\n[DRY RUN]")
        # Tính ước tính output duration (có thêm padding)
        pad_s = step_cfg["silence_pad_ms"] / 1000.0 * 2
        est_out_dur = total_input_dur + len(all_rows) * pad_s
        logger.info(f"  Sẽ xử lý    : {len(all_rows)} files")
        logger.info(f"  Est. output  : {format_duration(est_out_dur)}")
        logger.info(f"  Output dir   : {output_dir}")
        logger.info(f"  Wavs dir     : {wavs_dir}")

        # Phân phối duration input
        durations = [r["duration"] for r in all_rows]
        logger.info(f"  Duration: min={min(durations):.1f}s "
                    f"avg={np.mean(durations):.1f}s "
                    f"max={max(durations):.1f}s")
        sys.exit(0)

    # Khởi tạo processor
    processor = AudioProcessor(
        target_sr=step_cfg["sample_rate"],
        target_ch=step_cfg["channels"],
        target_lufs=step_cfg["target_lufs"],
        silence_pad_ms=step_cfg["silence_pad_ms"],
    )

    # ── Xử lý từng file ─────────────────────────────────────
    from tqdm import tqdm

    prefix      = step_cfg.get("output_prefix", "ngan")
    all_records: List[FormattedRecord] = []
    failed      = 0
    file_index  = 1   # Đánh số lại từ 1 cho toàn bộ dataset

    total_start = time.time()

    for row in tqdm(all_rows, desc="  Formatting", ncols=70):
        # Tên output: ngan_0001.wav, ngan_0002.wav, ...
        out_name = f"{prefix}_{file_index:05d}.wav"
        out_wav  = wavs_dir / out_name

        # Skip nếu đã có
        if args.skip_existing and out_wav.exists() and out_wav.stat().st_size > 0:
            try:
                import soundfile as sf
                info     = sf.info(str(out_wav))
                dur_out  = info.duration
                transcript = row["transcript"]
                all_records.append(FormattedRecord(
                    original_wav=row["wav_path"],
                    output_wav=out_wav,
                    transcript=transcript,
                    duration_in=row["duration"],
                    duration_out=dur_out,
                    source_file=row["source_file"],
                ))
                file_index += 1
                continue
            except Exception:
                pass  # Không đọc được → xử lý lại

        # Xử lý audio
        success, dur_out = processor.process(
            input_path=row["wav_path"],
            output_path=out_wav,
            logger=logger,
        )

        if not success:
            logger.warning(f"  ✗ Thất bại: {row['wav_path'].name}")
            failed += 1
            continue

        all_records.append(FormattedRecord(
            original_wav=row["wav_path"],
            output_wav=out_wav,
            transcript=row["transcript"],
            duration_in=row["duration"],
            duration_out=dur_out,
            source_file=row["source_file"],
        ))
        file_index += 1

    total_elapsed = time.time() - total_start
    logger.info(f"\n  Xử lý xong: {len(all_records)} OK | {failed} thất bại "
                f"({format_duration(total_elapsed)})")

    if not all_records:
        logger.error("Không có file nào được xử lý thành công!")
        sys.exit(1)

    # ── Viết filelists ───────────────────────────────────────
    logger.info("\n  Tạo filelists ...")
    n_train, n_val = write_filelists(
        records=all_records,
        output_dir=output_dir,
        val_ratio=val_ratio,
        encoding=step_cfg.get("filelist_encoding", "utf-8"),
        logger=logger,
    )

    # ── Dataset info JSON ────────────────────────────────────
    info = save_dataset_info(
        records=all_records,
        output_dir=output_dir,
        step_cfg=step_cfg,
        n_train=n_train,
        n_val=n_val,
        logger=logger,
    )

    # ── Verify output ────────────────────────────────────────
    verify_output_sample(all_records, n_samples=10, logger=logger)

    # ── Tổng kết ─────────────────────────────────────────────
    total_out_dur  = sum(r.duration_out for r in all_records)
    source_summary = info["sources"]

    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 7 — DATASET HOÀN CHỈNH")
    logger.info("=" * 60)
    logger.info(f"  Files đầu vào : {len(all_rows)}")
    logger.info(f"  Files đầu ra  : {len(all_records)}")
    logger.info(f"  Thất bại      : {failed}")
    logger.info(f"  Tổng duration : {format_duration(total_out_dur)}")
    logger.info(f"  Train / Val   : {n_train} / {n_val}")
    logger.info(f"  Thời gian     : {format_duration(total_elapsed)}")

    logger.info("\n  Nguồn audio:")
    for src, count in sorted(source_summary.items(), key=lambda x: -x[1]):
        logger.info(f"    {src}: {count} slices")

    logger.info(f"\n  Format: {step_cfg['sample_rate']}Hz / "
                f"mono / {step_cfg['bit_depth']}-bit PCM WAV")

    logger.info(f"\n  Output dataset: {output_dir}")
    logger.info(f"    {wavs_dir.name}/            ← {len(all_records)} file .wav")
    logger.info(f"    filelist_train.txt ← {n_train} records (StyleTTS2 input)")
    logger.info(f"    filelist_val.txt   ← {n_val}  records")
    logger.info(f"    dataset_info.json  ← metadata")

    # # StyleTTS2 readiness check
    # readiness = info["styletts2_readiness"]
    # if readiness["ready"]:
    #     logger.info(f"\n  {readiness['note']}")
    #     logger.info("\n  ──────────────────────────────────────────────────")
    #     logger.info("  PIPELINE HOÀN CHỈNH! Các bước tiếp theo:")
    #     logger.info("  ──────────────────────────────────────────────────")
    #     logger.info("  1. Clone StyleTTS2:")
    #     logger.info("       git clone https://github.com/yl4579/StyleTTS2")
    #     logger.info(f"  2. Copy dataset vào StyleTTS2/Data/:")
    #     logger.info(f"       cp -r {output_dir}/wavs StyleTTS2/Data/")
    #     logger.info(f"       cp {output_dir}/filelist_train.txt StyleTTS2/Data/")
    #     logger.info(f"       cp {output_dir}/filelist_val.txt   StyleTTS2/Data/")
    #     logger.info("  3. Chỉnh config StyleTTS2 (sample_rate: 24000)")
    #     logger.info("  4. Chạy fine-tuning:")
    #     logger.info("       python train_finetune.py --config Configs/config_ft.yml")
    # else:
    #     logger.warning(f"\n  {readiness['note']}")
    #     logger.warning("  Thêm file audio gốc vào raw_audio/ và chạy lại pipeline.")


if __name__ == "__main__":
    main()