"""
=============================================================
  BƯỚC 5: TRANSCRIPTION — Faster-Whisper
=============================================================
Mục tiêu: Chuyển đổi từng slice audio thành văn bản tiếng Việt,
          tạo các cặp (audio, text) chuẩn bị cho fine-tuning.

Input  : workdir/step04_filtered/  (filtered_manifest.csv + *.wav)
Output : workdir/step05_transcribed/
            <ten_file>/
                <ten_file>_0001.wav   ← copy từ bước 4
                <ten_file>_0001.txt   ← transcript tương ứng
            transcribed_manifest.csv  ← path|text (LJSpeech format)

Cách chạy:
  python step05_transcription.py
  python step05_transcription.py --source ten_file   # 1 source
  python step05_transcription.py --dry-run
  python step05_transcription.py --resume            # bỏ qua file đã có .txt

Lưu ý:
  - large-v3 + int8 cần ~2GB VRAM — phù hợp RTX 3050Ti
  - Nếu vẫn OOM → đổi whisper_model: "medium" trong config.yaml
  - Whisper tự detect ngôn ngữ nếu không set language, nhưng đặt "vi"
    giúp tránh nhầm sang tiếng Trung/Nhật với audio tiếng Việt
  - Các transcript rỗng hoặc < min_words từ sẽ bị loại (file audio vẫn giữ,
    chỉ không đưa vào manifest cuối)
=============================================================
"""
import os
import sys
import csv
import time
import shutil
import logging
import argparse
import re
import gc
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple

import yaml

#  CONFIG & LOGGING
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
    for lib in ["faster_whisper", "ctranslate2", "torch", "numba"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("step05")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"

#  DATA STRUCTURES
class TranscribedRecord(NamedTuple):
    source_file:  str
    slice_index:  int
    wav_path:     Path
    txt_path:     Path
    transcript:   str
    duration:     float
    avg_logprob:  float    # confidence score từ Whisper
    no_speech_prob: float  # xác suất không có tiếng nói

#  MANIFEST I/O
def load_filtered_manifest(
    manifest_path: Path,
    logger: logging.Logger,
) -> List[Dict]:
    """Đọc filtered_manifest.csv từ bước 4."""
    if not manifest_path.exists():
        logger.error(f"Không tìm thấy manifest: {manifest_path}")
        logger.error("Hãy chạy step04_speaker_filter.py trước.")
        sys.exit(1)

    rows = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav = Path(row["output_path"])
            if not wav.exists():
                logger.warning(f"  File không tồn tại, bỏ qua: {wav}")
                continue
            rows.append({
                "source_file":  row["source_file"],
                "slice_index":  int(row["slice_index"]),
                "duration":     float(row["duration_s"]),
                "wav_path":     wav,
            })

    logger.info(f"  Filtered manifest: {len(rows)} slices hợp lệ")
    return rows


def save_transcribed_manifest(
    records:      List[TranscribedRecord],
    output_path:  Path,
    filelist_fmt: str,
    logger:       logging.Logger,
):
    """
    Lưu 2 file:
    1. transcribed_manifest.csv — đầy đủ metadata (để debug)
    2. filelist.txt             — LJSpeech format "path|text" (cho StyleTTS2)
    """
    # Full CSV
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "slice_index", "duration_s",
            "avg_logprob", "no_speech_prob",
            "wav_path", "txt_path", "transcript",
        ])
        for r in records:
            writer.writerow([
                r.source_file, r.slice_index,
                f"{r.duration:.3f}",
                f"{r.avg_logprob:.4f}",
                f"{r.no_speech_prob:.4f}",
                str(r.wav_path),
                str(r.txt_path),
                r.transcript,
            ])
    logger.info(f"  Manifest CSV: {output_path} ({len(records)} records)")

    # LJSpeech filelist
    filelist_path = output_path.parent / "filelist.txt"
    with open(filelist_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(f"{r.wav_path}|{r.transcript}\n")
    logger.info(f"  LJSpeech filelist: {filelist_path}")

#  TEXT CLEANER
def clean_transcript(text: str) -> str:
    """
    Làm sạch transcript tiếng Việt từ Whisper.

    Xử lý:
    - Loại bỏ các tag đặc biệt của Whisper: [MUSIC], [Applause], (nhạc)...
    - Chuẩn hóa khoảng trắng
    - Loại bỏ ký tự đặc biệt không cần thiết
    - Giữ nguyên dấu câu tiếng Việt (., , ! ?)
    - KHÔNG sửa chính tả — để nguyên để model học đúng giọng đọc thực tế
    """
    if not text:
        return ""

    # Loại bỏ Whisper special tags: [MUSIC], [Noise], (nhạc nền)...
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\(.*?\)", "", text)

    # Loại bỏ ký tự đặc biệt không thuộc tiếng Việt
    # Giữ lại: chữ cái (bao gồm Unicode tiếng Việt), số, dấu câu cơ bản
    text = re.sub(r"[^\w\s\.,!?;:\-\u00C0-\u024F\u1E00-\u1EFF]", "", text)

    # Chuẩn hóa khoảng trắng
    text = re.sub(r"\s+", " ", text).strip()

    # Loại bỏ dấu câu lặp (... → ...)
    text = re.sub(r"\.{4,}", "...", text)
    text = re.sub(r"!{2,}", "!", text)
    text = re.sub(r"\?{2,}", "?", text)

    return text

def is_valid_transcript(
    text: str,
    min_words: int,
    logger: logging.Logger,
    wav_name: str = "",
) -> tuple[bool, str]:
    """
    Kiểm tra transcript có hợp lệ không.
    Returns: (is_valid, reason)
    """
    if not text or len(text.strip()) == 0:
        return False, "empty"

    words = text.strip().split()
    if len(words) < min_words:
        return False, f"too_few_words({len(words)}<{min_words})"

    # Kiểm tra transcript toàn ký tự không phải tiếng Việt
    # (thường là Whisper nhầm sang tiếng Anh/Trung)
    viet_pattern = re.compile(
        r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ"
        r"ÀÁẠẢÃÂẦẤẬẨẪĂẰẮẶẲẴÈÉẸẺẼÊỀẾỆỂỄÌÍỊỈĨÒÓỌỎÕÔỒỐỘỔỖƠỜỚỢỞỠÙÚỤỦŨƯỪỨỰỬỮỲÝỴỶỸĐ]",
        re.IGNORECASE
    )
    if not viet_pattern.search(text):
        return False, "no_vietnamese_chars"

    return True, "ok"

#  WHISPER ENGINE
class WhisperEngine:
    """
    Wrapper Faster-Whisper với lazy loading và VRAM management.
    """

    def __init__(
        self,
        model_size:   str   = "large-v3",
        compute_type: str   = "int8",
        device:       str   = "cuda",
        language:     str   = "vi",
        beam_size:    int   = 5,
    ):
        self.model_size   = model_size
        self.compute_type = compute_type
        self.device       = device
        self.language     = language
        self.beam_size    = beam_size
        self._model       = None

    def _load(self, logger: logging.Logger):
        if self._model is not None:
            return

        try:
            from faster_whisper import WhisperModel
        except ImportError:
            logger.error("faster-whisper chưa cài!")
            logger.error("Cài đặt: pip install faster-whisper")
            sys.exit(1)

        logger.info(
            f"  Load Faster-Whisper [{self.model_size}] "
            f"compute={self.compute_type} device={self.device} ..."
        )

        # Fallback sang CPU nếu CUDA không khả dụng
        device = self.device
        try:
            import torch
            if device == "cuda" and not torch.cuda.is_available():
                logger.warning("  CUDA không khả dụng → chạy trên CPU (chậm hơn)")
                device = "cpu"
                # CPU chỉ hỗ trợ int8 hoặc float32
                if self.compute_type == "float16":
                    self.compute_type = "int8"
        except ImportError:
            device = "cpu"

        try:
            self._model = WhisperModel(
                self.model_size,
                device=device,
                compute_type=self.compute_type,
                # download_root=None  # Dùng cache mặc định (~/.cache/huggingface)
            )
            logger.info(f"  Whisper loaded (device={device})")
        except Exception as e:
            logger.error(f"  Lỗi load Whisper: {e}")
            if "out of memory" in str(e).lower():
                logger.error("  VRAM không đủ. Thử: whisper_model: 'medium' trong config.yaml")
            raise

    def transcribe(
        self,
        wav_path:  Path,
        logger:    logging.Logger,
    ) -> tuple[str, float, float]:
        """
        Transcribe 1 file WAV.

        Returns:
            (transcript_text, avg_logprob, no_speech_prob)
            avg_logprob: confidence âm (thấp hơn = kém tin cậy hơn, thường < -1.0 là xấu)
            no_speech_prob: xác suất đây là im lặng/không phải tiếng nói (> 0.5 = đáng ngờ)
        """
        self._load(logger)

        try:
            segments, info = self._model.transcribe(
                str(wav_path),
                language=self.language,
                beam_size=self.beam_size,
                vad_filter=True,              # Lọc thêm im lặng trong file
                vad_parameters={
                    "min_silence_duration_ms": 500,
                    "speech_pad_ms": 200,
                },
                word_timestamps=False,         # Không cần word-level timestamps
                condition_on_previous_text=False,  # Tắt để tránh hallucination
                temperature=0.0,               # Greedy decoding — ổn định hơn
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
            )

            # Collect tất cả segments
            full_text   = ""
            avg_logprob = 0.0
            no_speech   = 0.0
            seg_count   = 0

            for seg in segments:
                full_text   += seg.text
                avg_logprob += seg.avg_logprob
                no_speech   += seg.no_speech_prob
                seg_count   += 1

            if seg_count > 0:
                avg_logprob /= seg_count
                no_speech   /= seg_count

            return full_text.strip(), avg_logprob, no_speech

        except Exception as e:
            logger.debug(f"  Lỗi transcribe {wav_path.name}: {e}")
            return "", -9.9, 1.0

    def release(self):
        self._model = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()

#  PROCESS 1 SOURCE GROUP
def process_source(
    source_name:  str,
    records:      List[Dict],
    output_dir:   Path,
    engine:       WhisperEngine,
    step_cfg:     dict,
    logger:       logging.Logger,
    resume:       bool = False,
) -> List[TranscribedRecord]:
    """
    Transcribe tất cả slice của 1 source file.
    """
    from tqdm import tqdm

    out_subdir = output_dir / source_name
    out_subdir.mkdir(parents=True, exist_ok=True)

    transcribed: List[TranscribedRecord] = []
    skipped = failed = filtered = 0

    for row in tqdm(records, desc=f"  {source_name}", ncols=70):
        src_wav   = row["wav_path"]
        slice_idx = row["slice_index"]
        duration  = row["duration"]

        # Đặt tên output
        out_wav = out_subdir / src_wav.name
        out_txt = out_subdir / (src_wav.stem + ".txt")

        # Resume: bỏ qua nếu đã có cả .wav và .txt
        if resume and out_txt.exists() and out_wav.exists():
            # Đọc lại transcript
            try:
                transcript = out_txt.read_text(encoding="utf-8").strip()
                transcribed.append(TranscribedRecord(
                    source_file=source_name,
                    slice_index=slice_idx,
                    wav_path=out_wav,
                    txt_path=out_txt,
                    transcript=transcript,
                    duration=duration,
                    avg_logprob=0.0,
                    no_speech_prob=0.0,
                ))
            except Exception:
                pass
            skipped += 1
            continue

        # Transcribe
        raw_text, avg_logprob, no_speech_prob = engine.transcribe(src_wav, logger)

        # Làm sạch text
        clean_text = clean_transcript(raw_text)

        # Kiểm tra chất lượng transcript
        valid, reason = is_valid_transcript(
            clean_text, step_cfg["min_words"], logger, src_wav.name
        )

        # Lọc thêm bằng Whisper confidence scores
        if valid:
            if no_speech_prob > 0.8:
                valid, reason = False, f"high_no_speech({no_speech_prob:.2f})"
            elif avg_logprob < -1.5:
                valid, reason = False, f"low_confidence({avg_logprob:.2f})"

        if not valid:
            if step_cfg.get("filter_empty_transcripts", True):
                logger.debug(f"  Loại {src_wav.name}: {reason}")
                filtered += 1
                continue
            # Nếu không filter → giữ lại nhưng text rỗng

        # Copy WAV sang output dir
        try:
            if not out_wav.exists() or out_wav.stat().st_size == 0:
                shutil.copy2(str(src_wav), str(out_wav))
        except Exception as e:
            logger.warning(f"  Lỗi copy {src_wav.name}: {e}")
            failed += 1
            continue

        # Lưu TXT
        try:
            out_txt.write_text(clean_text, encoding="utf-8")
        except Exception as e:
            logger.warning(f"  Lỗi lưu txt {out_txt.name}: {e}")
            failed += 1
            continue

        transcribed.append(TranscribedRecord(
            source_file=source_name,
            slice_index=slice_idx,
            wav_path=out_wav,
            txt_path=out_txt,
            transcript=clean_text,
            duration=duration,
            avg_logprob=avg_logprob,
            no_speech_prob=no_speech_prob,
        ))

    logger.info(
        f"  {source_name}: {len(transcribed)} OK | "
        f"{filtered} filtered | {skipped} skip | {failed} failed"
    )
    return transcribed

#  MAIN
def main():
    parser = argparse.ArgumentParser(
        description="Bước 5: Transcription bằng Faster-Whisper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",  default="config.yaml")
    parser.add_argument(
        "--source", default=None,
        help="Chỉ xử lý 1 source (tên stem, không đuôi .wav)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="In danh sách file, không chạy Whisper"
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Bỏ qua các slice đã có file .txt (tiếp tục sau khi bị ngắt)"
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg  = cfg["step05"]

    work_dir        = Path(paths_cfg["work_dir"])
    step04_dir      = work_dir / "step04_filtered"
    step_output_dir = work_dir / "step05_transcribed"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step05.log")

    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("  BƯỚC 5: TRANSCRIPTION (Faster-Whisper)")
    logger.info("=" * 60)
    logger.info(f"  Model      : {step_cfg['whisper_model']}")
    logger.info(f"  Compute    : {step_cfg['compute_type']} / {step_cfg['device']}")
    logger.info(f"  Language   : {step_cfg['language']}")
    logger.info(f"  Beam size  : {step_cfg['beam_size']}")
    logger.info(f"  Min words  : {step_cfg['min_words']}")
    logger.info(f"  Resume     : {args.resume}")
    logger.info(f"  Input      : {step04_dir}")
    logger.info(f"  Output     : {step_output_dir}")

    # Đọc manifest bước 4
    manifest_path = step04_dir / "filtered_manifest.csv"
    all_rows      = load_filtered_manifest(manifest_path, logger)

    if not all_rows:
        logger.error("Manifest rỗng. Kiểm tra lại bước 4.")
        sys.exit(1)

    # Lọc theo --source
    if args.source:
        all_rows = [r for r in all_rows if r["source_file"] == args.source]
        if not all_rows:
            avail = sorted(set(r["source_file"] for r in all_rows))
            logger.error(f"Không tìm thấy source '{args.source}'. Có sẵn: {avail}")
            sys.exit(1)
        logger.info(f"  Filter source: '{args.source}' — {len(all_rows)} slices")

    # Nhóm theo source
    source_groups: Dict[str, List[Dict]] = {}
    for row in all_rows:
        source_groups.setdefault(row["source_file"], []).append(row)

    total_slices = sum(len(v) for v in source_groups.values())
    total_dur    = sum(r["duration"] for r in all_rows)
    logger.info(
        f"  {total_slices} slices từ {len(source_groups)} source(s) "
        f"({format_duration(total_dur)})"
    )

    # Dry run
    if args.dry_run:
        logger.info("\n[DRY RUN]")
        for src, rows in source_groups.items():
            dur = sum(r["duration"] for r in rows)
            out_subdir = step_output_dir / src
            done = len(list(out_subdir.glob("*.txt"))) if out_subdir.exists() else 0
            logger.info(
                f"  {src}: {len(rows)} slices "
                f"({format_duration(dur)}) — {done} đã transcript"
            )
        sys.exit(0)

    # Khởi tạo Whisper engine
    engine = WhisperEngine(
        model_size=step_cfg["whisper_model"],
        compute_type=step_cfg["compute_type"],
        device=step_cfg["device"],
        language=step_cfg["language"],
        beam_size=step_cfg["beam_size"],
    )

    all_transcribed: List[TranscribedRecord] = []
    total_start = time.time()

    try:
        for src_name, rows in source_groups.items():
            logger.info(f"\n{'─'*50}")
            logger.info(
                f"  Source: {src_name} "
                f"({len(rows)} slices, {format_duration(sum(r['duration'] for r in rows))})"
            )
            t_start = time.time()

            records = process_source(
                source_name=src_name,
                records=rows,
                output_dir=step_output_dir,
                engine=engine,
                step_cfg=step_cfg,
                logger=logger,
                resume=args.resume,
            )

            all_transcribed.extend(records)

            # Thống kê confidence
            if records:
                logprobs = [r.avg_logprob for r in records if r.avg_logprob != 0.0]
                if logprobs:
                    logger.info(
                        f"  Confidence (avg_logprob): "
                        f"min={min(logprobs):.2f} "
                        f"avg={sum(logprobs)/len(logprobs):.2f} "
                        f"max={max(logprobs):.2f}"
                    )

            logger.info(f"  ✓ {src_name} xong ({format_duration(time.time() - t_start)})")

    finally:
        engine.release()

    # Lưu manifest
    if all_transcribed:
        manifest_out = step_output_dir / "transcribed_manifest.csv"
        save_transcribed_manifest(
            records=all_transcribed,
            output_path=manifest_out,
            filelist_fmt=cfg["step07"]["filelist_format"],
            logger=logger,
        )

    # Tổng kết
    total_elapsed = time.time() - total_start
    total_kept_dur = sum(r.duration for r in all_transcribed)

    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 5")
    logger.info("=" * 60)
    logger.info(f"  Input slices     : {total_slices}")
    logger.info(f"  Transcribed OK   : {len(all_transcribed)}")
    logger.info(f"  Filtered/dropped : {total_slices - len(all_transcribed)}")
    logger.info(f"  Tổng thời lượng  : {format_duration(total_kept_dur)}")
    logger.info(f"  Thời gian chạy   : {format_duration(total_elapsed)}")

    if total_slices > 0:
        rtf = total_elapsed / max(total_kept_dur, 1)
        logger.info(f"  RTF (real-time factor): {rtf:.2f}x")

    if all_transcribed:
        logger.info(f"\n  Output: {step_output_dir}")
        logger.info(f"  Filelist: {step_output_dir / 'filelist.txt'}")

    if total_kept_dur < 1800:
        logger.warning(
            f"\n⚠ Chỉ còn {format_duration(total_kept_dur)} sau khi lọc. "
            f"StyleTTS2 cần tối thiểu 30 phút."
        )

    logger.info("\n→ Chạy tiếp: python step06_quality_check.py")

if __name__ == "__main__":
    main()