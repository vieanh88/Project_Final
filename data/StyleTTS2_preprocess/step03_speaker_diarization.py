"""
=============================================================
  BƯỚC 3: DIARIZATION SLICING — pyannote/speaker-diarization-3.1
=============================================================
Mục tiêu: Phân đoạn người nói, xác định giọng Nguyễn Ngọc Ngạn
          (dominant speaker = tổng thời lượng lớn nhất), và chỉ
          cắt ra các sample THUẦN giọng Ngạn — không lẫn giọng nữ.

Thay đổi so với phiên bản VAD cũ:
  - Dùng pyannote/speaker-diarization-3.1 thay vì VAD thuần
  - Tự động phân biệt giọng Ngạn/giọng nữ ngay từ bước này
  - Chỉ export segment của dominant speaker (= giọng Ngạn)
  - Giọng nữ bị loại bỏ hoàn toàn trước khi sang bước 4
  - pyannote 3.1 nội bộ dùng wespeaker-ResNet34 → chính xác cao

Input  : workdir/step02_restored/  (*_restored.wav)
Output : workdir/step03_slices/
            <ten_file>/
                <ten_file>_0001.wav   ← chỉ chứa giọng Ngạn
                ...
            slices_manifest.csv
            diarization_rttm/         ← (tùy chọn, dùng --save-rttm)
                <ten_file>.rttm

Cách chạy:
  python step03_vad_slicing.py
  python step03_vad_slicing.py --input workdir/step02_restored/ten_file_restored.wav
  python step03_vad_slicing.py --dry-run
  python step03_vad_slicing.py --save-rttm    # lưu RTTM để debug diarization

Lưu ý:
  - Cần HF_TOKEN và accept terms cho:
      https://huggingface.co/pyannote/speaker-diarization-3.1
      https://huggingface.co/pyannote/segmentation-3.0
  - ~3-4GB VRAM; nếu thiếu VRAM → chạy trên CPU (chậm hơn ~8x)
  - Điều chỉnh max_speakers / merge_gap_s trong config.yaml nếu
    diarization nhận nhầm giọng
  - Dùng --save-rttm để kiểm tra kết quả phân đoạn bằng Audacity
    (File → Import → Annotations → chọn file .rttm)
=============================================================
"""
import os
import sys
import csv
import time
import logging
import argparse
import gc
import re
from pathlib import Path
from typing import List, Optional, Tuple, NamedTuple, Dict

import numpy as np
import yaml
from dotenv import load_dotenv

# ─── CONFIG & LOGGING ────────────────────────────────────────────────────────
def load_config(config_path: str = "config.yaml") -> dict:
    """Load config từ YAML, tự động thay thế biến môi trường dạng ${VAR}."""
    p = Path(config_path)
    if not p.exists():
        print(f"[LỖI] Không tìm thấy config: {config_path}")
        sys.exit(1)
    load_dotenv()
    with open(p, "r", encoding="utf-8") as f:
        content = f.read()

    def replace_env_var(m):
        return os.environ.get(m.group(1), m.group(0))

    content = re.sub(r"\${([^}]+)}", replace_env_var, content)
    return yaml.safe_load(content)

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
    for lib in ["pyannote", "pytorch_lightning", "speechbrain", "numba", "torch"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("step03")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"

# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────
class SpeakerSegment(NamedTuple):
    """Đoạn audio được diarization gán nhãn speaker."""
    speaker_id: str
    start: float   # giây
    end: float     # giây

    @property
    def duration(self) -> float:
        return self.end - self.start

class SpeechRegion(NamedTuple):
    """Vùng speech (sau khi đã lọc theo dominant speaker)."""
    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

class Slice(NamedTuple):
    """Một slice đã cắt và lưu ra file."""
    source_file: str
    slice_index: int
    start: float
    end: float
    output_path: str
    duration: float

# ─── HF TOKEN RESOLVER ────────────────────────────────────────────────────────
def resolve_hf_token(cfg_token: str) -> str:
    """Ưu tiên: config.yaml → env HF_TOKEN → ~/.huggingface/token."""
    if cfg_token and cfg_token.strip():
        return cfg_token.strip()
    env_token = os.environ.get("HF_TOKEN", "").strip()
    if env_token:
        return env_token
    hf_home = Path(os.environ.get("HF_HOME", Path.home() / ".huggingface"))
    token_file = hf_home / "token"
    if token_file.exists():
        token = token_file.read_text().strip()
        if token:
            return token
    return ""

# ─── DIARIZATION ENGINE ───────────────────────────────────────────────────────
class DiarizationEngine:
    """
    Wrapper pyannote/speaker-diarization-3.1.
    Phân đoạn người nói → trả về List[SpeakerSegment] sắp xếp theo thời gian.
    pyannote/speaker-diarization-3.1 nội bộ:
      - pyannote/segmentation-3.0 (VAD + speaker segmentation)
      - wespeaker-voxceleb-ResNet34 (speaker embedding)
      - Agglomerative clustering trên embedding
    """
    def __init__(
        self,
        hf_token: str,
        model_name: str = "pyannote/speaker-diarization-3.1",
        max_speakers: int = 5,
    ):
        self.hf_token = hf_token
        self.model_name = model_name
        self.max_speakers = max_speakers
        self._pipeline = None

    def _load(self, logger: logging.Logger):
        """Lazy load — chỉ load 1 lần để tiết kiệm thời gian."""
        if self._pipeline is not None:
            return

        if not self.hf_token:
            logger.error("Không tìm thấy HF_TOKEN!")
            logger.error("Cần accept terms tại:")
            logger.error("  https://huggingface.co/pyannote/speaker-diarization-3.1")
            logger.error("  https://huggingface.co/pyannote/segmentation-3.0")
            logger.error("Cài token: export HF_TOKEN=hf_xxx (Linux/macOS)")
            logger.error("           set HF_TOKEN=hf_xxx   (Windows)")
            sys.exit(1)

        try:
            from pyannote.audio import Pipeline
        except ImportError:
            logger.error("pyannote.audio chưa cài. Chạy: pip install pyannote.audio")
            sys.exit(1)

        logger.info(f"  Load {self.model_name} ...")
        try:
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token,
            )
        except Exception as e:
            logger.error(f"  Không load được model: {e}")
            logger.error("  Kiểm tra: (1) HF_TOKEN đúng, (2) đã accept terms chưa.")
            sys.exit(1)

        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self._pipeline.to(torch.device(device))
        logger.info(f"  Diarization pipeline loaded (device={device})")

    def diarize(
        self,
        audio_path: Path,
        logger: logging.Logger,
        rttm_output: Optional[Path] = None,
    ) -> List[SpeakerSegment]:
        """
        Chạy diarization trên 1 file audio.

        Args:
            audio_path:  Đường dẫn file WAV
            logger:      Logger instance
            rttm_output: Nếu có → lưu file RTTM để debug bằng Audacity

        Returns:
            Danh sách SpeakerSegment sắp xếp theo start time.
        """
        self._load(logger)
        logger.info(
            f"  Diarization: {audio_path.name} "
            f"(max_speakers={self.max_speakers}) ..."
        )

        try:
            result = self._pipeline(
                str(audio_path),
                max_speakers=self.max_speakers,
            )
        except Exception as e:
            logger.error(f"  Diarization thất bại: {e}")
            return []

        # Lưu RTTM để debug nếu cần
        if rttm_output:
            rttm_output.parent.mkdir(parents=True, exist_ok=True)
            with open(rttm_output, "w", encoding="utf-8") as f:
                result.write_rttm(f)
            logger.info(f"  RTTM đã lưu → {rttm_output.name}")

        # Chuyển pyannote Annotation → List[SpeakerSegment]
        segments: List[SpeakerSegment] = []
        for turn, _, speaker in result.itertracks(yield_label=True):
            segments.append(SpeakerSegment(
                speaker_id=speaker,
                start=turn.start,
                end=turn.end,
            ))

        segments.sort(key=lambda s: s.start)
        logger.info(f"  Raw segments: {len(segments)}")
        return segments

    def get_dominant_speaker(
        self,
        segments: List[SpeakerSegment],
        logger: logging.Logger,
    ) -> str:
        """
        Xác định speaker có tổng thời lượng lớn nhất = giọng Ngạn.

        Quy tắc này hoạt động tốt vì:
          - Ngạn là narrator → chiếm 60-70% trở lên thời lượng
          - Giọng nữ (host) chỉ chiếm 30-40% trở lên thời lượng
          - Tỉ lệ chênh lệch rõ ràng → không bị nhầm
        """
        if not segments:
            raise ValueError("Không có segment nào để xác định dominant speaker!")

        duration_map: Dict[str, float] = {}
        for seg in segments:
            duration_map[seg.speaker_id] = (
                duration_map.get(seg.speaker_id, 0.0) + seg.duration
            )

        total = sum(duration_map.values())
        dominant = max(duration_map, key=duration_map.get)

        logger.info("  Thống kê theo speaker:")
        for spk, dur in sorted(duration_map.items(), key=lambda x: -x[1]):
            pct = 100 * dur / total
            marker = " ← dominant (giọng Ngạn)" if spk == dominant else ""
            logger.info(f"    {spk}: {format_duration(dur)} ({pct:.1f}%){marker}")

        dominant_pct = 100 * duration_map[dominant] / total
        if dominant_pct < 40.0:
            logger.warning(
                f"  ⚠ Dominant speaker chỉ chiếm {dominant_pct:.1f}% "
                f"(< 40%) — thấp hơn kỳ vọng. "
                f"Thử tăng max_speakers hoặc kiểm tra file audio."
            )

        return dominant

    def release(self):
        """Giải phóng VRAM sau khi xong."""
        self._pipeline = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        gc.collect()
        logger_local = logging.getLogger("step03")
        logger_local.info("  Đã giải phóng VRAM (diarization pipeline).")

# ─── AUDIO SLICER ─────────────────────────────────────────────────────────────
class AudioSlicer:
    """
    Nhận danh sách segment của dominant speaker và cắt thành slice nhỏ.

    Luồng xử lý:
      1. Gộp segment gần nhau cùng dominant speaker (< merge_gap_s)
         → tránh tạo quá nhiều file cực ngắn khi Ngạn nói ngắt quãng
      2. Region <= max_duration → giữ nguyên
      3. Region > max_duration → tìm điểm im lặng gần nhất để cắt (đệ quy)
      4. Bỏ slice < min_duration
      5. Thêm padding đầu/cuối → lưu WAV
    """
    def __init__(
        self,
        min_duration: float,
        max_duration: float,
        merge_gap_s: float = 0.5,
        split_on_silence_ms: int = 800,
        padding_ms: int = 100,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.merge_gap_s = merge_gap_s
        self.split_on_silence_s = split_on_silence_ms / 1000.0
        self.padding_s = padding_ms / 1000.0

    def slice_dominant_segments(
        self,
        audio: np.ndarray,
        sample_rate: int,
        dominant_segments: List[SpeakerSegment],
        output_dir: Path,
        source_stem: str,
        logger: logging.Logger,
    ) -> List[Slice]:
        """
        Cắt audio theo segment của dominant speaker, lưu ra file WAV.

        Args:
            audio:             Toàn bộ audio file (numpy array)
            sample_rate:       Sample rate của audio
            dominant_segments: Chỉ segment của dominant speaker
            output_dir:        Thư mục output
            source_stem:       Tên file gốc (dùng để đặt tên slice)
            logger:            Logger instance

        Returns:
            Danh sách Slice đã lưu thành công.
        """
        try:
            import soundfile as sf
        except ImportError:
            logger.error("soundfile chưa cài. Chạy: pip install soundfile")
            sys.exit(1)

        output_dir.mkdir(parents=True, exist_ok=True)

        total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]
        total_duration = total_samples / sample_rate

        # Chuyển SpeakerSegment → SpeechRegion (chỉ cần start/end)
        regions = [SpeechRegion(s.start, s.end) for s in dominant_segments]

        # Gộp region gần nhau để tránh slice cực ngắn
        merged = self._merge_close_regions(regions)
        logger.info(
            f"  Regions sau merge (gap<{self.merge_gap_s}s): "
            f"{len(regions)} → {len(merged)}"
        )

        slices: List[Slice] = []
        slice_idx = 0

        for region in merged:
            # Chia region dài thành sub-segment <= max_duration
            sub_segments = self._split_region(audio, sample_rate, region)

            for seg_start, seg_end in sub_segments:
                duration = seg_end - seg_start

                # Bỏ qua nếu quá ngắn
                if duration < self.min_duration:
                    continue

                # Padding ở đầu và cuối (clamp về giới hạn file)
                padded_start = max(0.0, seg_start - self.padding_s)
                padded_end   = min(total_duration, seg_end + self.padding_s)

                start_sample = int(padded_start * sample_rate)
                end_sample   = int(padded_end * sample_rate)

                chunk = (
                    audio[start_sample:end_sample]
                    if audio.ndim == 1
                    else audio[start_sample:end_sample, :]
                )

                if len(chunk) == 0:
                    continue

                slice_idx += 1
                out_filename = f"{source_stem}_{slice_idx:04d}.wav"
                out_path = output_dir / out_filename

                try:
                    sf.write(str(out_path), chunk, sample_rate, subtype="PCM_16")
                except Exception as e:
                    logger.warning(f"  Lỗi lưu slice {slice_idx}: {e}")
                    continue

                slices.append(Slice(
                    source_file=source_stem,
                    slice_index=slice_idx,
                    start=padded_start,
                    end=padded_end,
                    output_path=str(out_path),
                    duration=padded_end - padded_start,
                ))

        return slices

    # ── Private helpers ───────────────────────────────────────────────────────

    def _merge_close_regions(
        self,
        regions: List[SpeechRegion],
    ) -> List[SpeechRegion]:
        """Gộp region cách nhau < merge_gap_s thành 1 region lớn hơn."""
        if not regions:
            return []
        merged = [SpeechRegion(regions[0].start, regions[0].end)]
        for r in regions[1:]:
            last = merged[-1]
            if r.start - last.end <= self.merge_gap_s:
                merged[-1] = SpeechRegion(last.start, r.end)
            else:
                merged.append(SpeechRegion(r.start, r.end))
        return merged

    def _split_region(
        self,
        audio: np.ndarray,
        sample_rate: int,
        region: SpeechRegion,
    ) -> List[Tuple[float, float]]:
        """
        Chia 1 region thành các sub-segment <= max_duration.

        Chiến lược:
          - Nếu region <= max_duration → trả về nguyên vẹn
          - Nếu region > max_duration → tìm điểm im lặng gần nhất
            trong cửa sổ [max_duration - window, max_duration] → cắt → đệ quy
        """
        if region.duration <= self.max_duration:
            return [(region.start, region.end)]

        # Tìm điểm im lặng để cắt
        search_window_s = min(self.split_on_silence_s * 2, 1.0)
        cut_search_start = region.start + self.max_duration - search_window_s
        cut_search_end   = region.start + self.max_duration

        best_cut = self._find_best_cut(
            audio, sample_rate, cut_search_start, cut_search_end
        )

        # Fallback: cắt cứng tại max_duration nếu không tìm được điểm im lặng
        if best_cut is None:
            best_cut = region.start + self.max_duration

        first_seg = (region.start, best_cut)
        remainder = SpeechRegion(best_cut, region.end)

        return [first_seg] + self._split_region(audio, sample_rate, remainder)

    def _find_best_cut(
        self,
        audio: np.ndarray,
        sample_rate: int,
        search_start_s: float,
        search_end_s: float,
        frame_ms: float = 10.0,
    ) -> Optional[float]:
        """
        Tìm điểm im lặng (RMS thấp nhất) trong khoảng [search_start_s, search_end_s].
        Returns: timestamp (giây) hoặc None nếu không tìm được.
        """
        total_samples = len(audio) if audio.ndim == 1 else audio.shape[0]
        start_sample = max(0, int(search_start_s * sample_rate))
        end_sample   = min(total_samples, int(search_end_s * sample_rate))

        if end_sample <= start_sample:
            return None

        frame_samples = int(frame_ms / 1000.0 * sample_rate)
        if frame_samples == 0:
            return None

        # Lấy channel 0 nếu multichannel
        segment = (
            audio[start_sample:end_sample]
            if audio.ndim == 1
            else audio[start_sample:end_sample, 0]
        )

        n_frames = len(segment) // frame_samples
        if n_frames == 0:
            return None

        rms_values = [
            np.sqrt(np.mean(
                segment[i * frame_samples:(i + 1) * frame_samples]
                .astype(np.float64) ** 2
            ))
            for i in range(n_frames)
        ]

        best_frame = int(np.argmin(rms_values))
        return search_start_s + (best_frame + 0.5) * frame_ms / 1000.0

# ─── MANIFEST ─────────────────────────────────────────────────────────────────
def save_manifest(
    all_slices: List[Slice],
    manifest_path: Path,
    logger: logging.Logger,
):
    """Lưu danh sách tất cả slice ra file CSV (tương thích với step04)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "slice_index",
            "start_s", "end_s", "duration_s",
            "output_path",
        ])
        for s in all_slices:
            writer.writerow([
                s.source_file, s.slice_index,
                f"{s.start:.3f}", f"{s.end:.3f}", f"{s.duration:.3f}",
                s.output_path,
            ])
    logger.info(f"  Manifest: {manifest_path} ({len(all_slices)} slices)")

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bước 3: Diarization-based slicing (pyannote 3.1)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Đường dẫn file config (mặc định: config.yaml)")
    parser.add_argument("--input", default=None,
                        help="Xử lý 1 file cụ thể (*_restored.wav từ bước 2)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Liệt kê file sẽ xử lý, không chạy diarization")
    parser.add_argument("--save-rttm", action="store_true",
                        help="Lưu file RTTM (debug diarization trong Audacity)")
    args = parser.parse_args()

    cfg       = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg  = cfg["step03"]

    work_dir        = Path(paths_cfg["work_dir"])
    step02_dir      = work_dir / "step02_restored"
    step_output_dir = work_dir / "step03_speaker_diarization"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step03.log")

    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("  BƯỚC 3: DIARIZATION SLICING (pyannote 3.1)")
    logger.info("=" * 60)
    logger.info(f"  Model      : {step_cfg['diarization_model']}")
    logger.info(f"  MaxSpk     : {step_cfg['max_speakers']}")
    logger.info(f"  MergeGap   : {step_cfg['merge_gap_s']}s")
    logger.info(f"  Min/Max    : {step_cfg['min_duration_s']}s / {step_cfg['max_duration_s']}s")
    logger.info(f"  Padding    : {step_cfg['segment_padding_ms']}ms")
    logger.info(f"  Save RTTM  : {args.save_rttm}")
    logger.info(f"  Input      : {step02_dir}")
    logger.info(f"  Output     : {step_output_dir}")

    # ── Lấy danh sách file input ──────────────────────────────────────────────
    if args.input:
        p = Path(args.input)
        if not p.exists():
            logger.error(f"File không tồn tại: {args.input}")
            sys.exit(1)
        input_files = [p]
    else:
        if not step02_dir.exists():
            logger.error(f"Thư mục bước 2 không tồn tại: {step02_dir}")
            logger.error("Hãy chạy step02_audio_restoration.py trước.")
            sys.exit(1)
        input_files = sorted(step02_dir.glob("*_restored.wav"))
        if not input_files:
            logger.error(f"Không tìm thấy *_restored.wav trong {step02_dir}")
            sys.exit(1)

    logger.info(f"  Files      : {len(input_files)} file")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("\n[DRY RUN]")
        for i, f in enumerate(input_files, 1):
            stem = f.stem.replace("_restored", "")
            out_subdir = step_output_dir / stem
            existing = list(out_subdir.glob("*.wav")) if out_subdir.exists() else []
            status = f"đã có {len(existing)} slices" if existing else "chưa xử lý"
            logger.info(f"  {i:3d}. {f.name} → {stem}/ [{status}]")
        sys.exit(0)

    # ── Khởi tạo engines ──────────────────────────────────────────────────────
    hf_token = resolve_hf_token(step_cfg.get("hf_token", ""))

    diarizer = DiarizationEngine(
        hf_token=hf_token,
        model_name=step_cfg["diarization_model"],
        max_speakers=step_cfg["max_speakers"],
    )
    slicer = AudioSlicer(
        min_duration=step_cfg["min_duration_s"],
        max_duration=step_cfg["max_duration_s"],
        merge_gap_s=step_cfg["merge_gap_s"],
        split_on_silence_ms=step_cfg["split_on_silence_ms"],
        padding_ms=step_cfg["segment_padding_ms"],
    )

    try:
        import soundfile as sf
    except ImportError:
        logger.error("soundfile chưa cài. Chạy: pip install soundfile")
        sys.exit(1)

    # Thư mục lưu RTTM debug (tùy chọn)
    rttm_base = step_output_dir / "diarization_rttm" if args.save_rttm else None

    all_slices: List[Slice] = []
    results = {"success": [], "skipped": [], "failed": []}
    total_start = time.time()

    # ── Processing loop ────────────────────────────────────────────────────────
    for idx, input_file in enumerate(input_files, 1):
        stem = input_file.stem
        if stem.endswith("_restored"):
            stem = stem[:-len("_restored")]

        out_subdir = step_output_dir / stem
        logger.info(f"\n[{idx}/{len(input_files)}] {input_file.name}")

        # Cache check
        if step_cfg["skip_existing"] and out_subdir.exists():
            existing = sorted(out_subdir.glob("*.wav"))
            if existing:
                logger.info(f"  Skip (cache): {len(existing)} slices đã có")
                for i, wav in enumerate(existing, 1):
                    try:
                        info = sf.info(str(wav))
                        dur  = info.duration
                    except Exception:
                        dur = 0.0
                    all_slices.append(Slice(
                        source_file=stem, slice_index=i,
                        start=0.0, end=dur,
                        output_path=str(wav), duration=dur,
                    ))
                results["skipped"].append(input_file.name)
                continue

        t_start = time.time()

        # Đọc toàn bộ audio vào RAM (cần để slice theo sample index)
        try:
            audio, sr = sf.read(str(input_file), dtype="float32", always_2d=False)
        except Exception as e:
            logger.error(f"  Lỗi đọc file: {e}")
            results["failed"].append(input_file.name)
            continue

        total_dur = (len(audio) if audio.ndim == 1 else audio.shape[0]) / sr
        logger.info(f"  Audio: {sr}Hz, {format_duration(total_dur)}")

        # ── Diarization ───────────────────────────────────────────────────────
        rttm_path = (rttm_base / f"{stem}.rttm") if rttm_base else None
        segments = diarizer.diarize(input_file, logger, rttm_output=rttm_path)

        if not segments:
            logger.warning("  Diarization không trả về segment nào! Bỏ qua file.")
            logger.warning("  Gợi ý: Kiểm tra file audio có bị corrupt không.")
            results["failed"].append(input_file.name)
            continue

        # ── Xác định dominant speaker (= giọng Ngạn) ─────────────────────────
        try:
            dominant = diarizer.get_dominant_speaker(segments, logger)
        except ValueError as e:
            logger.error(f"  {e}")
            results["failed"].append(input_file.name)
            continue

        dominant_segs = [s for s in segments if s.speaker_id == dominant]
        non_dominant  = [s for s in segments if s.speaker_id != dominant]

        total_dominant_dur  = sum(s.duration for s in dominant_segs)
        total_excluded_dur  = sum(s.duration for s in non_dominant)

        logger.info(
            f"  Giữ lại (giọng Ngạn): {len(dominant_segs)} segments, "
            f"{format_duration(total_dominant_dur)}"
        )
        logger.info(
            f"  Loại bỏ (giọng khác): {len(non_dominant)} segments, "
            f"{format_duration(total_excluded_dur)}"
        )

        # ── Cắt và lưu slice ──────────────────────────────────────────────────
        slices = slicer.slice_dominant_segments(
            audio=audio,
            sample_rate=sr,
            dominant_segments=dominant_segs,
            output_dir=out_subdir,
            source_stem=stem,
            logger=logger,
        )

        elapsed = time.time() - t_start

        if not slices:
            logger.warning("  Không tạo được slice nào!")
            logger.warning(f"  Gợi ý: Thử giảm min_duration_s (hiện tại: {step_cfg['min_duration_s']}s)")
            results["failed"].append(input_file.name)
            continue

        durations = [s.duration for s in slices]
        logger.info(
            f"  ✓ {len(slices)} slices | "
            f"Duration: min={min(durations):.1f}s "
            f"avg={np.mean(durations):.1f}s "
            f"max={max(durations):.1f}s | "
            f"Elapsed: {format_duration(elapsed)}"
        )

        all_slices.extend(slices)
        results["success"].append(input_file.name)

    # ── Giải phóng VRAM ────────────────────────────────────────────────────────
    diarizer.release()

    # ── Lưu manifest ──────────────────────────────────────────────────────────
    manifest_path = step_output_dir / "slices_manifest.csv"
    save_manifest(all_slices, manifest_path, logger)

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 3")
    logger.info("=" * 60)
    logger.info(f"  ✓ Thành công : {len(results['success'])} file")
    logger.info(f"  ⏭ Bỏ qua    : {len(results['skipped'])} file (cache)")
    logger.info(f"  ✗ Thất bại  : {len(results['failed'])} file")
    logger.info(f"  Tổng slices  : {len(all_slices)}")

    total_dur_all = sum(s.duration for s in all_slices)
    logger.info(f"  Tổng thời lượng: {format_duration(total_dur_all)}")
    logger.info(f"  Thời gian chạy : {format_duration(total_elapsed)}")

    if results["failed"]:
        logger.warning("\nFile thất bại:")
        for f in results["failed"]:
            logger.warning(f"  - {f}")
        logger.warning("\nGợi ý xử lý lỗi:")
        logger.warning("  - Diarization lỗi    → kiểm tra HF_TOKEN và accept terms")
        logger.warning("  - Nhận nhầm speaker  → điều chỉnh max_speakers trong config")
        logger.warning("  - Dominant quá thấp  → chạy --save-rttm để debug bằng Audacity")
        logger.warning("  - Mất nhiều slice    → giảm min_duration_s trong config")

    logger.info(f"\n  Manifest: {manifest_path}")
    if args.save_rttm and rttm_base:
        logger.info(f"  RTTM files: {rttm_base}")
    logger.info("\n→ Chạy tiếp: python step04_speaker_filter.py")

if __name__ == "__main__":
    main()