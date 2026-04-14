"""
=============================================================
  BƯỚC 6: QUALITY CHECK — DNSMOS P.808
=============================================================
Mục tiêu: Đánh giá chất lượng âm thanh từng slice bằng mô hình
          DNSMOS (Deep Noise Suppression MOS) của Microsoft.
          Chỉ giữ lại các đoạn có Overall MOS > ngưỡng (mặc định 3.5).

Thang điểm DNSMOS (1-5):
  SIG  — Signal quality  (chất lượng giọng nói)
  BAK  — Background noise (độ sạch nền)
  OVRL — Overall MOS      (tổng hợp — tiêu chí chính)

Input  : workdir/step05_transcribed/  (transcribed_manifest.csv)
Output : workdir/step06_quality/
            <ten_file>/
                <ten_file>_0001.wav   ← các slice pass
                <ten_file>_0001.txt   ← transcript tương ứng
            quality_manifest.csv      ← manifest cuối với MOS scores
            quality_scores.csv        ← tất cả scores (pass + fail, để debug)
            filelist.txt              ← LJSpeech format, chỉ slice pass

Cách chạy:
  python step06_quality_check.py
  python step06_quality_check.py --source ten_file
  python step06_quality_check.py --dry-run
  python step06_quality_check.py --score-only   # chỉ chấm điểm, không copy file

Lưu ý:
  - DNSMOS chạy trên CPU bằng ONNX — không cần GPU, rất nhẹ
  - Model DNSMOS sẽ tự download lần đầu (~10MB)
  - Nếu muốn download thủ công:
      https://github.com/microsoft/DNS-Challenge (thư mục DNSMOS)
      Đặt 2 file .onnx vào thư mục dnsmos_models/ cùng cấp với script
=============================================================
"""

from email.mime import audio
import os
import sys
import csv
import time
import shutil
import logging
import argparse
import urllib.request
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple, Tuple

import librosa
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
    return logging.getLogger("step06")


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"


# ============================================================
#  DATA STRUCTURES
# ============================================================

class MosScore(NamedTuple):
    sig:  float   # Signal quality
    bak:  float   # Background noise quality
    ovrl: float   # Overall MOS


class QualityRecord(NamedTuple):
    source_file:  str
    slice_index:  int
    wav_path:     Path
    txt_path:     Optional[Path]
    transcript:   str
    duration:     float
    mos:          MosScore
    passed:       bool


# ============================================================
#  MANIFEST I/O
# ============================================================

def load_transcribed_manifest(
    manifest_path: Path,
    logger: logging.Logger,
) -> List[Dict]:
    """Đọc transcribed_manifest.csv từ bước 5."""
    if not manifest_path.exists():
        logger.error(f"Không tìm thấy manifest: {manifest_path}")
        logger.error("Hãy chạy step05_transcription.py trước.")
        sys.exit(1)

    rows = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav = Path(row["wav_path"])
            if not wav.exists():
                logger.warning(f"  WAV không tồn tại, bỏ qua: {wav.name}")
                continue
            txt = Path(row["txt_path"]) if row.get("txt_path") else None
            rows.append({
                "source_file":  row["source_file"],
                "slice_index":  int(row["slice_index"]),
                "duration":     float(row["duration_s"]),
                "wav_path":     wav,
                "txt_path":     txt,
                "transcript":   row.get("transcript", ""),
            })

    logger.info(f"  Manifest: {len(rows)} records")
    return rows


def save_quality_manifests(
    records:     List[QualityRecord],
    output_dir:  Path,
    logger:      logging.Logger,
):
    """
    Lưu 3 file:
    1. quality_scores.csv   — tất cả scores (pass + fail), để phân tích
    2. quality_manifest.csv — chỉ các slice PASS, đầy đủ metadata
    3. filelist.txt         — LJSpeech format cho StyleTTS2
    """
    # ── 1. All scores (pass + fail) ──────────────────────────
    scores_path = output_dir / "quality_scores.csv"
    with open(scores_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "slice_index", "duration_s",
            "mos_sig", "mos_bak", "mos_ovrl",
            "passed", "wav_path", "transcript",
        ])
        for r in records:
            writer.writerow([
                r.source_file, r.slice_index, f"{r.duration:.3f}",
                f"{r.mos.sig:.3f}", f"{r.mos.bak:.3f}", f"{r.mos.ovrl:.3f}",
                r.passed,
                str(r.wav_path), r.transcript,
            ])
    logger.info(f"  All scores: {scores_path} ({len(records)} records)")

    # ── 2. Quality manifest (pass only) ─────────────────────
    passed = [r for r in records if r.passed]
    manifest_path = output_dir / "quality_manifest.csv"
    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "slice_index", "duration_s",
            "mos_sig", "mos_bak", "mos_ovrl",
            "wav_path", "txt_path", "transcript",
        ])
        for r in passed:
            writer.writerow([
                r.source_file, r.slice_index, f"{r.duration:.3f}",
                f"{r.mos.sig:.3f}", f"{r.mos.bak:.3f}", f"{r.mos.ovrl:.3f}",
                str(r.wav_path),
                str(r.txt_path) if r.txt_path else "",
                r.transcript,
            ])
    logger.info(f"  Quality manifest: {manifest_path} ({len(passed)} passed)")

    # ── 3. LJSpeech filelist ─────────────────────────────────
    filelist_path = output_dir / "filelist.txt"
    with open(filelist_path, "w", encoding="utf-8") as f:
        for r in passed:
            f.write(f"{r.wav_path}|{r.transcript}\n")
    logger.info(f"  Filelist: {filelist_path}")

    return passed


# ============================================================
#  DNSMOS MODEL MANAGER
# ============================================================

# URLs model DNSMOS từ Microsoft DNS-Challenge repo
DNSMOS_MODELS = {
    "sig_bak_ovr": {
        "url": (
            "https://github.com/microsoft/DNS-Challenge/raw/master/"
            "DNSMOS/DNSMOS/sig_bak_ovr.onnx"
        ),
        "filename": "sig_bak_ovr.onnx",
    },
    "p808_onnx": {
        "url": (
            "https://github.com/microsoft/DNS-Challenge/raw/master/"
            "DNSMOS/DNSMOS/model_v8.onnx"
        ),
        "filename": "model_v8.onnx",
    },
}

DNSMOS_SR        = 16000   # Model expect 16kHz
DNSMOS_WIN_LEN   = 0.02    # 20ms window
DNSMOS_HOP_LEN   = 0.01    # 10ms hop
DNSMOS_FFT_SIZE  = 320
DNSMOS_N_MELS    = 120


def ensure_dnsmos_models(model_dir: Path, logger: logging.Logger) -> Dict[str, Path]:
    """
    Đảm bảo các file ONNX đã có. Download nếu chưa có.
    Returns dict {"sig_bak_ovr": Path, "p808_onnx": Path}
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    for key, info in DNSMOS_MODELS.items():
        dest = model_dir / info["filename"]
        if dest.exists() and dest.stat().st_size > 1000:
            logger.info(f"  DNSMOS model cache: {dest.name}")
            paths[key] = dest
            continue

        logger.info(f"  Download DNSMOS model: {info['filename']} ...")
        try:
            urllib.request.urlretrieve(info["url"], str(dest))
            logger.info(f"  Download xong: {dest.name} ({dest.stat().st_size/1024:.0f} KB)")
            paths[key] = dest
        except Exception as e:
            logger.error(f"  Download thất bại: {e}")
            logger.error(
                f"  Download thủ công tại:\n"
                f"    {info['url']}\n"
                f"  Đặt vào: {dest}"
            )
            sys.exit(1)

    return paths


# ============================================================
#  DNSMOS SCORER
# ============================================================

class DNSMOSScorer:
    """
    DNSMOS P.808 scorer dùng ONNX Runtime.
    Dựa trên implementation gốc của Microsoft DNS-Challenge.
    Chạy trên CPU — nhẹ, không cần GPU.
    """

    INPUT_LENGTH = 9.01   # giây — độ dài input cố định của model

    def __init__(self, model_paths: Dict[str, Path], use_gpu: bool = False):
        self.model_paths = model_paths
        self.use_gpu     = use_gpu
        self._session_p808   = None
        self._session_sigbak = None

    def _load(self, logger: logging.Logger):
        if self._session_p808 is not None:
            return

        try:
            import onnxruntime as ort
        except ImportError:
            logger.error("onnxruntime chưa cài!")
            logger.error("Cài đặt: pip install onnxruntime")
            sys.exit(1)

        providers = ["CPUExecutionProvider"]
        if self.use_gpu:
            try:
                import torch
                if torch.cuda.is_available():
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            except ImportError:
                pass

        opts = ort.SessionOptions()
        opts.log_severity_level = 3  # Tắt verbose log

        self._session_p808 = ort.InferenceSession(
            str(self.model_paths["p808_onnx"]),
            sess_options=opts,
            providers=providers,
        )
        self._session_sigbak = ort.InferenceSession(
            str(self.model_paths["sig_bak_ovr"]),
            sess_options=opts,
            providers=providers,
        )
        logger.info(f"  DNSMOS models loaded (providers={providers})")

    def score(
        self,
        wav_path: Path,
        logger:   logging.Logger,
    ) -> Optional[MosScore]:
        """
        Tính DNSMOS score cho 1 file WAV.
        Returns MosScore hoặc None nếu lỗi.
        """
        self._load(logger)

        try:
            audio, sr = self._load_audio_16k(wav_path)
        except Exception as e:
            # Đổi từ debug sang error để thấy lỗi nếu có
            logger.error(f"  Lỗi đọc audio {wav_path.name}: {e}")
            return None

        if len(audio) < sr * 0.5:  # Quá ngắn
            return None

        try:
            # Tính log mel spectrogram
            input_features = self._compute_features(audio, sr)

            if input_features is None:
                return None

            # Chạy model P.808 (overall MOS)
            p808_input_name = self._session_p808.get_inputs()[0].name
            p808_out = self._session_p808.run(
                None,
                {p808_input_name: input_features["p808_feats"]},
            )
            
            # Chạy model SIG/BAK
            sigbak_input_name = self._session_sigbak.get_inputs()[0].name
            sigbak_out = self._session_sigbak.run(
                None,
                {sigbak_input_name: input_features["sigbak_feats"]},
            )
            
            # === TRÍCH XUẤT ĐIỂM ===
            p808_flat = np.array(p808_out).flatten()
            sigbak_flat = np.array(sigbak_out).flatten()
            
            # Raw outputs — chưa calibrate
            mos_ovrl_raw = float(p808_flat[0])
            mos_sig_raw  = float(sigbak_flat[0])   # [SIG, BAK, OVRL]
            mos_bak_raw  = float(sigbak_flat[1])

            # ← THÊM BƯỚC NÀY: Polynomial calibration (thiếu bước này = điểm sai)
            mos_sig, mos_bak, mos_ovrl = self._get_polyfit_val(
                mos_sig_raw, mos_bak_raw, mos_ovrl_raw
            )

            # Clip về range hợp lệ
            mos_sig  = float(np.clip(mos_sig,  1.0, 5.0))
            mos_bak  = float(np.clip(mos_bak,  1.0, 5.0))
            mos_ovrl = float(np.clip(mos_ovrl, 1.0, 5.0))

            return MosScore(sig=mos_sig, bak=mos_bak, ovrl=mos_ovrl)

        except Exception as e:
            # QUAN TRỌNG: Đổi từ logger.debug thành logger.error 
            # để file log không bao giờ giấu lỗi của bạn nữa!
            logger.error(f"  Lỗi DNSMOS inference {wav_path.name}: {e}")
            return None

    def _load_audio_16k(self, wav_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio và resample về 16kHz mono."""
        try:
            import soundfile as sf
            audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        except Exception:
            # Fallback sang librosa nếu soundfile thất bại
            import librosa
            audio, sr = librosa.load(str(wav_path), sr=None, mono=True)

        # Convert sang mono
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)

        # Resample về 16kHz nếu cần
        if sr != DNSMOS_SR:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=DNSMOS_SR)
            except ImportError:
                # Fallback: simple resample bằng numpy
                ratio      = DNSMOS_SR / sr
                new_length = int(len(audio) * ratio)
                indices    = np.linspace(0, len(audio) - 1, new_length)
                audio      = np.interp(indices, np.arange(len(audio)), audio)

        return audio.astype(np.float32), DNSMOS_SR

    def _compute_features(
        self,
        audio: np.ndarray,
        sr:    int,
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Chuẩn bị input cho 2 model theo đúng spec Microsoft DNSMOS.
        Ref: github.com/microsoft/DNS-Challenge/blob/master/DNSMOS/dnsmos_local.py
        """
        import librosa

        target_len = int(self.INPUT_LENGTH * sr)  # 9.01 * 16000 = 144160
        if len(audio) < target_len:
            audio = np.pad(audio, (0, target_len - len(audio)))
        else:
            audio = audio[:target_len]

        # ── P.808 feature: dùng audio[:-160] như Microsoft gốc ──
        audio_for_mel = audio[:-160]  # bỏ 160 sample cuối — theo spec gốc
        mel_spec = librosa.feature.melspectrogram(
            y=audio_for_mel,
            sr=sr,
            n_fft=321,         # frame_size + 1 = 320 + 1, theo Microsoft
            hop_length=160,
            n_mels=120,
        )
        # Công thức chuẩn Microsoft: (power_to_db + 40) / 40
        mel_db = (librosa.power_to_db(mel_spec, ref=np.max) + 40) / 40
        p808_feats = mel_db.T.astype(np.float32)[np.newaxis, :, :]  # (1, frames, 120)

        # ── SIG/BAK feature: raw audio ──
        sigbak_feats = audio[np.newaxis, :].astype(np.float32)  # (1, 144160)

        return {"p808_feats": p808_feats, "sigbak_feats": sigbak_feats}
    
    @staticmethod
    def _get_polyfit_val(sig_raw: float, bak_raw: float, ovr_raw: float):
        """
        Polynomial calibration — BẮT BUỘC theo Microsoft DNSMOS.
        Chuyển raw model output → MOS score thực sự trên thang 1-5.
        """
        p_ovr = np.poly1d([-0.06766283, 1.11546468, 0.04602535])
        p_sig = np.poly1d([-0.08397278, 1.22083953, 0.0052439])
        p_bak = np.poly1d([-0.13166888, 1.60915514, -0.39604546])
        return float(p_sig(sig_raw)), float(p_bak(bak_raw)), float(p_ovr(ovr_raw))

    @staticmethod
    def _mel_filterbank(
        n_mels: int,
        n_fft:  int,
        sr:     int,
        fmin:   float,
        fmax:   float,
    ) -> np.ndarray:
        """Tạo mel filterbank matrix (n_mels, n_fft//2+1)."""

        def hz_to_mel(f):
            return 2595 * np.log10(1 + f / 700)

        def mel_to_hz(m):
            return 700 * (10 ** (m / 2595) - 1)

        n_freqs  = n_fft // 2 + 1
        freqs    = np.linspace(0, sr / 2, n_freqs)
        mel_min  = hz_to_mel(fmin)
        mel_max  = hz_to_mel(fmax)
        mel_pts  = np.linspace(mel_min, mel_max, n_mels + 2)
        hz_pts   = mel_to_hz(mel_pts)

        fb = np.zeros((n_mels, n_freqs), dtype=np.float32)
        for m in range(n_mels):
            f_lo  = hz_pts[m]
            f_mid = hz_pts[m + 1]
            f_hi  = hz_pts[m + 2]
            for k, f in enumerate(freqs):
                if f_lo <= f <= f_mid and f_mid > f_lo:
                    fb[m, k] = (f - f_lo) / (f_mid - f_lo)
                elif f_mid < f <= f_hi and f_hi > f_mid:
                    fb[m, k] = (f_hi - f) / (f_hi - f_mid)

        return fb

    def release(self):
        self._session_p808   = None
        self._session_sigbak = None
        import gc
        gc.collect()


# ============================================================
#  STATISTICS PRINTER
# ============================================================

def print_mos_distribution(
    records: List[QualityRecord],
    threshold: float,
    logger: logging.Logger,
):
    """In phân phối MOS score để người dùng hiểu rõ chất lượng dataset."""
    if not records:
        return

    ovrl_scores = [r.mos.ovrl for r in records]
    sig_scores  = [r.mos.sig  for r in records]
    bak_scores  = [r.mos.bak  for r in records]

    def stats(arr):
        return (
            f"min={min(arr):.2f} "
            f"p25={np.percentile(arr, 25):.2f} "
            f"med={np.median(arr):.2f} "
            f"p75={np.percentile(arr, 75):.2f} "
            f"max={max(arr):.2f}"
        )

    logger.info(f"  OVRL: {stats(ovrl_scores)}")
    logger.info(f"  SIG : {stats(sig_scores)}")
    logger.info(f"  BAK : {stats(bak_scores)}")

    # Histogram đơn giản
    bins = [1.0, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.01]
    bin_labels = ["1.0-2.0", "2.0-2.5", "2.5-3.0", "3.0-3.5",
                  "3.5-4.0", "4.0-4.5", "4.5-5.0"]
    logger.info("  Phân phối OVRL MOS:")
    for i, label in enumerate(bin_labels):
        count = sum(1 for s in ovrl_scores if bins[i] <= s < bins[i + 1])
        pct   = 100 * count / len(ovrl_scores)
        bar   = "█" * int(pct / 2)
        marker = " ← ngưỡng" if bins[i] <= threshold < bins[i + 1] else ""
        logger.info(f"    {label}: {bar} {count} ({pct:.1f}%){marker}")


# ============================================================
#  MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Bước 6: Quality check bằng DNSMOS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config",     default="config.yaml")
    parser.add_argument(
        "--source", default=None,
        help="Chỉ xử lý 1 source (tên stem)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="In thống kê, không chạy DNSMOS"
    )
    parser.add_argument(
        "--score-only", action="store_true",
        help="Chỉ chấm điểm và lưu CSV, không copy file sang output dir"
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg  = cfg["step06"]

    work_dir        = Path(paths_cfg["work_dir"])
    step05_dir      = work_dir / "step05_transcribed"
    step_output_dir = work_dir / "step06_quality"
    model_dir       = Path("dnsmos_models")   # Thư mục lưu model ONNX
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step06.log")

    logger = setup_logging(log_path)

    threshold_ovrl = step_cfg["min_overall_mos"]
    threshold_sig  = step_cfg.get("min_sig_mos",  0.0)
    threshold_bak  = step_cfg.get("min_bak_mos",  0.0)

    logger.info("=" * 60)
    logger.info("  BƯỚC 6: QUALITY CHECK (DNSMOS P.808)")
    logger.info("=" * 60)
    logger.info(f"  Ngưỡng OVRL: > {threshold_ovrl}")
    logger.info(f"  Ngưỡng SIG : > {threshold_sig}  (0 = bỏ qua)")
    logger.info(f"  Ngưỡng BAK : > {threshold_bak}  (0 = bỏ qua)")
    logger.info(f"  Save CSV   : {step_cfg['save_score_csv']}")
    logger.info(f"  Input      : {step05_dir}")
    logger.info(f"  Output     : {step_output_dir}")

    # Đọc manifest bước 5
    manifest_path = step05_dir / "transcribed_manifest.csv"
    all_rows      = load_transcribed_manifest(manifest_path, logger)

    if not all_rows:
        logger.error("Manifest rỗng. Kiểm tra lại bước 5.")
        sys.exit(1)

    # Lọc theo --source
    if args.source:
        all_rows = [r for r in all_rows if r["source_file"] == args.source]
        if not all_rows:
            logger.error(f"Không tìm thấy source '{args.source}'")
            sys.exit(1)
        logger.info(f"  Filter source: '{args.source}' — {len(all_rows)} records")

    # Nhóm theo source
    source_groups: Dict[str, List[Dict]] = {}
    for row in all_rows:
        source_groups.setdefault(row["source_file"], []).append(row)

    total_dur = sum(r["duration"] for r in all_rows)
    logger.info(
        f"  {len(all_rows)} slices từ {len(source_groups)} source(s) "
        f"({format_duration(total_dur)})"
    )

    # Dry run
    if args.dry_run:
        logger.info("\n[DRY RUN]")
        for src, rows in source_groups.items():
            dur = sum(r["duration"] for r in rows)
            logger.info(f"  {src}: {len(rows)} slices ({format_duration(dur)})")
        logger.info(f"\n  Sẽ chấm điểm {len(all_rows)} slices bằng DNSMOS P.808")
        sys.exit(0)

    # Download DNSMOS models nếu chưa có
    logger.info("\n  Kiểm tra DNSMOS models ...")
    model_paths = ensure_dnsmos_models(model_dir, logger)

    # Khởi tạo scorer
    scorer = DNSMOSScorer(model_paths=model_paths, use_gpu=step_cfg.get("use_gpu", False))

    all_records:  List[QualityRecord] = []
    total_start = time.time()

    try:
        from tqdm import tqdm

        for src_name, rows in source_groups.items():
            logger.info(f"\n{'─'*50}")
            logger.info(
                f"  Source: {src_name} "
                f"({len(rows)} slices, {format_duration(sum(r['duration'] for r in rows))})"
            )
            t_start    = time.time()
            src_records: List[QualityRecord] = []

            for row in tqdm(rows, desc=f"  DNSMOS {src_name}", ncols=70):
                wav_path  = row["wav_path"]
                txt_path  = row.get("txt_path")
                transcript = row.get("transcript", "")
                duration   = row["duration"]

                mos = scorer.score(wav_path, logger)

                if mos is None:
                    # Không tính được score → gán score thấp
                    mos    = MosScore(sig=1.0, bak=1.0, ovrl=1.0)
                    passed = False
                else:
                    passed = (
                        mos.ovrl > threshold_ovrl
                        and (threshold_sig == 0 or mos.sig > threshold_sig)
                        and (threshold_bak == 0 or mos.bak > threshold_bak)
                    )

                src_records.append(QualityRecord(
                    source_file=src_name,
                    slice_index=row["slice_index"],
                    wav_path=wav_path,
                    txt_path=Path(txt_path) if txt_path else None,
                    transcript=transcript,
                    duration=duration,
                    mos=mos,
                    passed=passed,
                ))

            # Thống kê source này
            passed_src = [r for r in src_records if r.passed]
            passed_dur = sum(r.duration for r in passed_src)
            logger.info(
                f"  {src_name}: {len(passed_src)}/{len(src_records)} pass "
                f"({format_duration(passed_dur)})"
            )
            logger.info("  Phân phối MOS:")
            print_mos_distribution(src_records, threshold_ovrl, logger)

            # Copy file pass sang output dir (nếu không --score-only)
            if not args.score_only:
                out_subdir = step_output_dir / src_name
                out_subdir.mkdir(parents=True, exist_ok=True)

                updated_records: List[QualityRecord] = []
                for r in src_records:
                    if not r.passed:
                        updated_records.append(r)
                        continue

                    # Copy WAV
                    dest_wav = out_subdir / r.wav_path.name
                    try:
                        if not dest_wav.exists():
                            shutil.copy2(str(r.wav_path), str(dest_wav))
                    except Exception as e:
                        logger.warning(f"  Lỗi copy WAV {r.wav_path.name}: {e}")
                        updated_records.append(QualityRecord(
                            r.source_file, r.slice_index,
                            r.wav_path, r.txt_path,
                            r.transcript, r.duration, r.mos, False,
                        ))
                        continue

                    # Copy TXT nếu có
                    dest_txt = None
                    if r.txt_path and r.txt_path.exists():
                        dest_txt = out_subdir / r.txt_path.name
                        try:
                            if not dest_txt.exists():
                                shutil.copy2(str(r.txt_path), str(dest_txt))
                        except Exception as e:
                            logger.warning(f"  Lỗi copy TXT {r.txt_path.name}: {e}")

                    updated_records.append(QualityRecord(
                        r.source_file, r.slice_index,
                        dest_wav,
                        dest_txt,
                        r.transcript, r.duration, r.mos, True,
                    ))

                src_records = updated_records

            all_records.extend(src_records)
            logger.info(f"  ✓ {src_name} ({format_duration(time.time() - t_start)})")

    finally:
        scorer.release()

    # Lưu manifests
    if step_cfg["save_score_csv"] or True:
        passed_records = save_quality_manifests(all_records, step_output_dir, logger)

    # Tổng kết
    total_elapsed  = time.time() - total_start
    passed_all     = [r for r in all_records if r.passed]
    rejected_all   = [r for r in all_records if not r.passed]
    passed_dur     = sum(r.duration for r in passed_all)
    rejected_dur   = sum(r.duration for r in rejected_all)

    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 6")
    logger.info("=" * 60)
    logger.info(f"  Input slices   : {len(all_records)}")
    logger.info(f"  ✓ Pass         : {len(passed_all)} ({format_duration(passed_dur)})")
    logger.info(f"  ✗ Reject       : {len(rejected_all)} ({format_duration(rejected_dur)})")
    logger.info(f"  Tỉ lệ giữ lại : {100*len(passed_all)/max(len(all_records),1):.1f}%")
    logger.info(f"  Thời gian chạy : {format_duration(total_elapsed)}")

    if all_records:
        logger.info("\n  Thống kê MOS tổng hợp:")
        print_mos_distribution(all_records, threshold_ovrl, logger)

    if passed_dur < 1800:
        logger.warning(
            f"\n⚠ Chỉ còn {format_duration(passed_dur)} sau quality check. "
            f"StyleTTS2 cần tối thiểu 30 phút.\n"
            f"  Gợi ý: Giảm ngưỡng min_overall_mos: {threshold_ovrl} → {threshold_ovrl - 0.2:.1f}"
        )
    else:
        logger.info(
            f"\n✅ {format_duration(passed_dur)} — Dataset đủ chất lượng cho fine-tuning!"
        )

    logger.info(f"\n  Output: {step_output_dir}")
    logger.info(f"  Score CSV: {step_output_dir / 'quality_scores.csv'}")
    logger.info("\n→ Chạy tiếp: python step07_formatting.py")


if __name__ == "__main__":
    main()