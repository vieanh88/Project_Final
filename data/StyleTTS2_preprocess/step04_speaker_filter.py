"""
=============================================================
  BƯỚC 4: SPEAKER FILTER — pyannote/embedding (ECAPA-TDNN)
=============================================================
Mục tiêu: Kiểm tra lại chất lượng diarization từ bước 3 bằng
          cosine similarity — loại bỏ bất kỳ slice nào còn sót
          lẫn giọng không phải Ngạn (diarization errors).

Thay đổi so với phiên bản cũ:
  - Bỏ HDBSCAN clustering (bước 3 đã diarize → tất cả slice
    đã từ dominant speaker → không cần cluster để tìm giọng Ngạn)
  - Bootstrap đơn giản: N slice dài nhất → tính centroid reference
    (slice từ bước 3 đã sạch → reference chuẩn, không bị nhiễu)
  - pyannote/embedding verify thuần: cosine similarity với centroid
  - Kết quả: giữ lại slice có similarity >= threshold

Strategy:
  1. Đọc slices từ bước 3 (đã là dominant speaker segment)
  2. Tính pyannote/embedding cho từng slice
  3. Xây reference: centroid của N slice dài nhất
  4. Verify: cosine similarity >= threshold → pass
  5. Copy slice pass sang step04_filtered/

Input  : workdir/step03_speaker_diarization/  (slices_manifest.csv + *.wav)
Output : workdir/step04_filtered/
            <ten_file>/
                <ten_file>_0001.wav
                ...
            filtered_manifest.csv

Cách chạy:
  python step04_speaker_filter.py
  python step04_speaker_filter.py --source ten_file   # chỉ xử lý 1 nguồn
  python step04_speaker_filter.py --dry-run
  python step04_speaker_filter.py --inspect           # in similarity scores

Lưu ý:
  - Điều chỉnh similarity_threshold trong config.yaml nếu:
      * Bỏ sót giọng Ngạn  → giảm xuống 0.60
      * Còn lẫn giọng nữ   → tăng lên 0.75
  - Dùng --inspect để xem phân phối similarity trước khi quyết định threshold
=============================================================
"""
import os
import sys
import csv
import time
import shutil
import logging
import argparse
import gc
from pathlib import Path
from typing import List, Optional, Dict, NamedTuple

import numpy as np
import yaml

# Pyannote và Wespeaker đều dùng torchaudio để load audio, nên import ở đây để tránh lỗi khi cài thiếu torchaudio
import torchaudio
import torch
import tempfile, soundfile as sf # File lưu tạm cho chuyển đổi

# Đảm bảo HF_API_KEY đã được load từ .env (nếu có)
from dotenv import load_dotenv

# ─── CONFIG & LOGGING ────────────────────────────────────────────────────────
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
    for lib in ["torch", "numba", "speechbrain", "pyannote"]:
        logging.getLogger(lib).setLevel(logging.WARNING)
    return logging.getLogger("step04")

def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    return f"{int(seconds // 60)}m {seconds % 60:.0f}s"

# ─── DATA STRUCTURES ──────────────────────────────────────────────────────────
class SliceRecord(NamedTuple):
    source_file: str
    slice_index: int
    start: float
    end: float
    duration: float
    wav_path: Path

class VerifiedSlice(NamedTuple):
    record: SliceRecord
    embedding: np.ndarray
    similarity: float
    passed: bool

# ─── MANIFEST I/O ─────────────────────────────────────────────────────────────
def load_manifest(
    manifest_path: Path,
    logger: logging.Logger,
) -> List[SliceRecord]:
    """Đọc slices_manifest.csv từ bước 3."""
    if not manifest_path.exists():
        logger.error(f"Không tìm thấy manifest: {manifest_path}")
        logger.error("Hãy chạy step03_vad_slicing.py trước.")
        sys.exit(1)

    records: List[SliceRecord] = []
    with open(manifest_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            wav_path = Path(row["output_path"])
            if not wav_path.exists():
                logger.warning(f"  File không tồn tại, bỏ qua: {wav_path.name}")
                continue
            records.append(SliceRecord(
                source_file=row["source_file"],
                slice_index=int(row["slice_index"]),
                start=float(row["start_s"]),
                end=float(row["end_s"]),
                duration=float(row["duration_s"]),
                wav_path=wav_path,
            ))

    logger.info(f"  Đọc manifest: {len(records)} slices hợp lệ")
    return records

def save_filtered_manifest(
    verified: List[VerifiedSlice],
    output_path: Path,
    logger: logging.Logger,
):
    """Lưu filtered_manifest.csv cho bước 5."""
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "source_file", "slice_index",
            "start_s", "end_s", "duration_s",
            "similarity", "output_path",
        ])
        for v in verified:
            if v.passed:
                writer.writerow([
                    v.record.source_file,
                    v.record.slice_index,
                    f"{v.record.start:.3f}",
                    f"{v.record.end:.3f}",
                    f"{v.record.duration:.3f}",
                    f"{v.similarity:.4f}",
                    str(v.record.wav_path),
                ])
    passed = sum(1 for v in verified if v.passed)
    logger.info(f"  Filtered manifest: {output_path} ({passed} slices)")

# ─── pyannote/embedding ENGINE ─────────────────────────────────────────────────
class PyannoteEmbeddingEngine:
    """
    Wrapper dùng pyannote/embedding (ECAPA-TDNN) để trích xuất speaker embedding.
    Nhẹ, nhanh và là lớp kiểm tra chéo (cross-check) tuyệt vời cho bước 3.
    """

    def __init__(self, model_name: str = "pyannote/embedding", use_gpu: bool = True):
        self.model_name = model_name
        self.use_gpu = use_gpu
        self._inference = None

    def _load(self, logger: logging.Logger):
        if self._inference is not None:
            return

        logger.info(f"  Load Pyannote Embedding model: {self.model_name} ...")
        
        # Lấy HF Token từ biến môi trường (Giống cách làm ở step03)
        load_dotenv()  # Đảm bảo .env được load nếu chưa
        hf_token = os.getenv("HF_API_KEY")
        if not hf_token:
            logger.error("Không tìm thấy HF_API_KEY hoặc HF_TOKEN trong biến môi trường!")
            sys.exit(1)

        try:
            from pyannote.audio import Model
            from pyannote.audio import Inference
            import torch
            
            # === BẢN VÁ LỖI TRIỆT ĐỂ CHO PYTORCH 2.6 ===
            # Lưu lại hàm load gốc của PyTorch
            original_torch_load = torch.load
            
            # Tạo một hàm load giả (đã được nới lỏng bảo mật)
            def _trusted_load(*args, **kwargs):
                kwargs['weights_only'] = False
                return original_torch_load(*args, **kwargs)
                
            try:
                # Ghi đè hàm load thành hàm giả
                torch.load = _trusted_load
                # Tải model từ HuggingFace (Lúc này máy quét PyTorch 2.6 đã bị tắt)
                model = Model.from_pretrained(self.model_name, use_auth_token=hf_token)
            finally:
                # QUAN TRỌNG: Ngay sau khi tải xong, trả lại hàm load gốc 
                # để đảm bảo an toàn cho toàn bộ phần còn lại của pipeline
                torch.load = original_torch_load
            # ============================================
            
            # Setup device
            device = torch.device("cuda" if self.use_gpu and torch.cuda.is_available() else "cpu")
            model.to(device)

            # Khởi tạo pipeline inference
            self._inference = Inference(model, window="whole")
            logger.info(f"  Pyannote Embedding loaded trên thiết bị: {device}")

        except ImportError:
            logger.error("pyannote.audio chưa cài. Hãy chạy: pip install pyannote.audio")
            sys.exit(1)
        except Exception as e:
            logger.error(f"  Lỗi load Pyannote Embedding: {e}")
            raise

    def extract_embedding(
        self,
        wav_path: Path,
        logger: logging.Logger,
    ) -> Optional[np.ndarray]:
        """
        Trích xuất L2-normalized embedding cho 1 file WAV.
        """
        self._load(logger)
        try:
            # Pyannote inference tự động xử lý sample rate (resample về 16kHz nội bộ)
            # Không cần tạo file temporary như Wespeaker
            emb = self._inference(str(wav_path))

            # Chuyển đổi về numpy array 1D
            if hasattr(emb, "data"):
                emb = emb.data
            emb = np.array(emb, dtype=np.float32).flatten()

            # Chuẩn hóa L2 (L2-normalization) để dùng Cosine Similarity
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return emb / norm

        except Exception as e:
            logger.warning(f"  Lỗi extract embedding {wav_path.name}: {e}")
            return None

    def release(self):
        self._inference = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        import gc
        gc.collect()

# ─── BOOTSTRAP: XÂY REFERENCE EMBEDDING ──────────────────────────────────────
def build_reference_embedding(
    records: List[SliceRecord],
    embeddings: Dict[str, np.ndarray],
    n_bootstrap: int,
    min_duration: float,
    logger: logging.Logger,
) -> np.ndarray:
    """
    Xây dựng reference embedding của giọng Ngạn từ N slice dài nhất.

    Tại sao đơn giản hóa được (không cần HDBSCAN nữa):
      Bước 3 đã diarize và chỉ giữ lại segment của dominant speaker.
      → Tất cả slice trong manifest đều là giọng Ngạn (hoặc rất gần đó).
      → Reference centroid từ N slice dài nhất đã đủ chính xác.
      → HDBSCAN clustering chỉ cần thiết khi input là hỗn hợp nhiều speaker,
        trường hợp này không còn sau khi step 3 lọc sẵn.

    Ưu điểm của chiến lược "N slice dài nhất":
      - Slice dài hơn → ít bị nhiễu từ boundary diarization
      - Slice dài hơn → embedding ổn định hơn (nhiều context hơn)
      - Tổng coverage tốt → centroid đại diện cho nhiều ngữ cảnh giọng Ngạn

    Args:
        records:     Danh sách SliceRecord từ bước 3
        embeddings:  Dict[wav_path_str → embedding array]
        n_bootstrap: Số slice dùng để xây reference
        min_duration: Thời lượng tối thiểu để dùng làm reference (giây)
        logger:      Logger instance

    Returns:
        L2-normalized centroid embedding.
    Raises:
        ValueError nếu không có đủ embedding.
    """
    # Lọc slice có embedding và đủ dài
    candidates = [
        r for r in records
        if str(r.wav_path) in embeddings and r.duration >= min_duration
    ]

    if not candidates:
        # Relaxed fallback: bỏ điều kiện min_duration
        candidates = [r for r in records if str(r.wav_path) in embeddings]
        if candidates:
            logger.warning(
                f"  Không có slice nào >= {min_duration}s "
                f"→ bỏ điều kiện duration để xây reference"
            )

    if not candidates:
        raise ValueError("Không có embedding nào để xây reference!")

    # Chọn N slice dài nhất
    candidates_sorted = sorted(candidates, key=lambda r: r.duration, reverse=True)
    selected = candidates_sorted[:n_bootstrap]
    total_ref_dur = sum(r.duration for r in selected)

    logger.info(
        f"  Reference: {len(selected)} slice dài nhất "
        f"(tổng: {format_duration(total_ref_dur)}, "
        f"avg: {total_ref_dur/len(selected):.1f}s)"
    )

    # Tính centroid embedding
    embs = np.stack([embeddings[str(r.wav_path)] for r in selected])
    centroid = np.mean(embs, axis=0)
    norm = np.linalg.norm(centroid)
    return centroid / norm if norm > 1e-8 else centroid

# ─── VERIFY SLICES ────────────────────────────────────────────────────────────
def verify_slices(
    records: List[SliceRecord],
    embeddings: Dict[str, np.ndarray],
    reference_emb: np.ndarray,
    threshold: float,
    logger: logging.Logger,
) -> List[VerifiedSlice]:
    """
    Tính cosine similarity giữa từng slice và reference embedding.
    Slice có similarity >= threshold → passed = True.

    Args:
        records:       Tất cả slice cần verify
        embeddings:    Dict[wav_path_str → embedding]
        reference_emb: Centroid embedding của giọng Ngạn
        threshold:     Ngưỡng similarity (config: similarity_threshold)
        logger:        Logger instance

    Returns:
        Danh sách VerifiedSlice (có cả passed và rejected).
    """
    results: List[VerifiedSlice] = []

    for r in records:
        key = str(r.wav_path)
        emb = embeddings.get(key)

        if emb is None:
            # Không trích xuất được embedding → reject an toàn
            results.append(VerifiedSlice(r, np.zeros(1), 0.0, False))
            continue

        # Cosine similarity = dot product của 2 vector đã L2-normalize
        similarity = float(np.dot(emb, reference_emb))
        passed     = similarity >= threshold

        results.append(VerifiedSlice(r, emb, similarity, passed))

    passed_count  = sum(1 for v in results if v.passed)
    rejected_count = len(results) - passed_count
    logger.info(
        f"  Verify xong: {passed_count} pass / {rejected_count} reject "
        f"(threshold={threshold})"
    )
    return results

# ─── COPY FILTERED FILES ──────────────────────────────────────────────────────
def copy_passed_slices(
    verified: List[VerifiedSlice],
    output_dir: Path,
    logger: logging.Logger,
) -> List[VerifiedSlice]:
    """
    Copy slice đã pass sang step04_filtered/<source>/.
    Cập nhật wav_path trong record sang đường dẫn mới.
    """
    updated: List[VerifiedSlice] = []
    copy_errors = 0

    for v in verified:
        if not v.passed:
            updated.append(v)
            continue

        src      = v.record.wav_path
        dest_dir = output_dir / v.record.source_file
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / src.name

        try:
            shutil.copy2(str(src), str(dest))
        except Exception as e:
            logger.warning(f"  Lỗi copy {src.name}: {e}")
            copy_errors += 1
            updated.append(VerifiedSlice(v.record, v.embedding, v.similarity, False))
            continue

        # Tạo record mới với path đã cập nhật
        new_record = SliceRecord(
            source_file=v.record.source_file,
            slice_index=v.record.slice_index,
            start=v.record.start,
            end=v.record.end,
            duration=v.record.duration,
            wav_path=dest,
        )
        updated.append(VerifiedSlice(new_record, v.embedding, v.similarity, True))

    if copy_errors:
        logger.warning(f"  {copy_errors} file bị lỗi khi copy")

    return updated

# ─── MAIN ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Bước 4: Speaker filter bằng Pyannote embedding/ECAPA-TDNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--source", default=None,
        help="Chỉ xử lý 1 source file (tên stem, không có đuôi .wav)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Chỉ đọc manifest và in thống kê, không chạy embedding"
    )
    parser.add_argument(
        "--inspect", action="store_true",
        help="In similarity score của tất cả slices (hữu ích để chỉnh threshold)"
    )
    args = parser.parse_args()

    cfg       = load_config(args.config)
    paths_cfg = cfg["paths"]
    step_cfg  = cfg["step04"]

    work_dir        = Path(paths_cfg["work_dir"])
    step03_dir      = work_dir / "step03_speaker_diarization"
    step_output_dir = work_dir / "step04_filtered"
    step_output_dir.mkdir(parents=True, exist_ok=True)
    log_path = str(work_dir / "logs" / "step04.log")

    logger = setup_logging(log_path)

    logger.info("=" * 60)
    logger.info("  BƯỚC 4: SPEAKER FILTER (Pyannote embedding/ECAPA-TDNN)")
    logger.info("=" * 60)
    logger.info(f"  Model      : {step_cfg['pyannote_embedding_model']}")
    logger.info(f"  Threshold  : {step_cfg['similarity_threshold']}")
    logger.info(f"  Bootstrap N: {step_cfg['n_bootstrap_segments']}")
    logger.info(f"  Min ref dur: {step_cfg['min_bootstrap_duration_s']}s")
    logger.info(f"  Input      : {step03_dir}")
    logger.info(f"  Output     : {step_output_dir}")
    logger.info(f"  [INFO] HDBSCAN đã bỏ — bước 3 đã diarize sẵn")

    # ── Đọc manifest bước 3 ───────────────────────────────────────────────────
    manifest_path = step03_dir / "slices_manifest.csv"
    all_records   = load_manifest(manifest_path, logger)

    if not all_records:
        logger.error("Manifest rỗng. Kiểm tra lại bước 3.")
        sys.exit(1)

    # Lọc theo --source nếu có
    if args.source:
        filtered_records = [r for r in all_records if r.source_file == args.source]
        if not filtered_records:
            avail = sorted(set(r.source_file for r in all_records))
            logger.error(f"Không tìm thấy source '{args.source}'")
            logger.error(f"Có sẵn: {avail}")
            sys.exit(1)
        all_records = filtered_records
        logger.info(f"  Filter source: '{args.source}' — {len(all_records)} slices")

    # Nhóm theo source file
    source_groups: Dict[str, List[SliceRecord]] = {}
    for r in all_records:
        source_groups.setdefault(r.source_file, []).append(r)

    logger.info(f"  {len(all_records)} slices từ {len(source_groups)} source file(s)")

    # ── Dry run ───────────────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("\n[DRY RUN]")
        for src, records in source_groups.items():
            total_dur = sum(r.duration for r in records)
            logger.info(
                f"  {src}: {len(records)} slices, "
                f"{format_duration(total_dur)}"
            )
        sys.exit(0)

    # ── Kiểm tra reference audio ngoài (nếu có) ───────────────────────────────
    ref_audio_path = paths_cfg.get("reference_audio", "").strip()
    external_reference = Path(ref_audio_path) if ref_audio_path else None
    if external_reference and not external_reference.exists():
        logger.warning(f"  reference_audio không tồn tại: {external_reference}")
        logger.warning("  → Dùng bootstrap từ N slice dài nhất")
        external_reference = None

    # ── Khởi tạo PyannoteEmbeddingEngine ─────────────────────────────────────────────
    engine = PyannoteEmbeddingEngine(
        model_name="pyannote/embedding",
        use_gpu=True,
    )

    all_verified: List[VerifiedSlice] = []
    results = {"success": [], "skipped": [], "failed": []}
    total_start = time.time()

    # Xây reference từ external audio nếu có (1 lần, dùng chung cho mọi source)
    external_ref_emb: Optional[np.ndarray] = None
    if external_reference:
        logger.info(f"\n  Trích xuất embedding từ reference audio: {external_reference.name}")
        external_ref_emb = engine.extract_embedding(external_reference, logger)
        if external_ref_emb is None:
            logger.warning("  Không trích xuất được → dùng bootstrap")

    try:
        # ── Xử lý từng source file ────────────────────────────────────────────
        for src_name, records in source_groups.items():
            logger.info(f"\n{'─' * 50}")
            logger.info(f"  Source: {src_name} ({len(records)} slices)")

            out_subdir = step_output_dir / src_name

            # Cache check
            if step_cfg["skip_existing"] and out_subdir.exists():
                existing = list(out_subdir.glob("*.wav"))
                if existing:
                    logger.info(f"  Skip (cache): {len(existing)} slices đã có")
                    results["skipped"].append(src_name)
                    for wav in sorted(existing):
                        try:
                            import soundfile as sf
                            info = sf.info(str(wav))
                            dur  = info.duration
                        except Exception:
                            dur = 0.0
                        r = SliceRecord(
                            source_file=src_name, slice_index=0,
                            start=0, end=dur, duration=dur, wav_path=wav,
                        )
                        all_verified.append(VerifiedSlice(r, np.zeros(1), 1.0, True))
                    continue

            t_start = time.time()

            # ── 4a: Trích xuất embedding cho tất cả slices ────────────────────
            logger.info(f"  Trích xuất embeddings ({len(records)} slices) ...")
            embeddings: Dict[str, np.ndarray] = {}
            failed_emb = 0

            from tqdm import tqdm
            for r in tqdm(records, desc=f"  Embedding {src_name[:20]}", ncols=70):
                emb = engine.extract_embedding(r.wav_path, logger)
                if emb is not None:
                    embeddings[str(r.wav_path)] = emb
                else:
                    failed_emb += 1

            logger.info(
                f"  Embeddings: {len(embeddings)}/{len(records)} "
                f"({failed_emb} thất bại)"
            )

            if len(embeddings) < 5:
                logger.error(
                    f"  Quá ít embedding ({len(embeddings)}) — bỏ qua source này."
                )
                results["failed"].append(src_name)
                continue

            # ── 4b: Xây dựng reference embedding ─────────────────────────────
            if external_ref_emb is not None:
                logger.info("  Dùng external reference audio")
                reference_emb = external_ref_emb
            else:
                logger.info("  Bootstrap: tính centroid từ N slice dài nhất ...")
                try:
                    reference_emb = build_reference_embedding(
                        records=records,
                        embeddings=embeddings,
                        n_bootstrap=step_cfg["n_bootstrap_segments"],
                        min_duration=step_cfg["min_bootstrap_duration_s"],
                        logger=logger,
                    )
                except ValueError as e:
                    logger.error(f"  Bootstrap thất bại: {e}")
                    results["failed"].append(src_name)
                    continue

            # ── 4c: Verify tất cả slices ──────────────────────────────────────
            logger.info(
                f"  Verify similarity (threshold={step_cfg['similarity_threshold']}) ..."
            )
            verified = verify_slices(
                records=records,
                embeddings=embeddings,
                reference_emb=reference_emb,
                threshold=step_cfg["similarity_threshold"],
                logger=logger,
            )

            # In similarity scores nếu --inspect
            if args.inspect:
                logger.info("\n  [INSPECT] Similarity scores (sorted):")
                for v in sorted(verified, key=lambda v: v.similarity, reverse=True):
                    status = "✓" if v.passed else "✗"
                    logger.info(
                        f"    {status} {v.record.wav_path.name}: "
                        f"{v.similarity:.4f}  ({v.record.duration:.1f}s)"
                    )
                logger.info("")

            # Thống kê
            passed   = [v for v in verified if v.passed]
            rejected = [v for v in verified if not v.passed]
            passed_dur   = sum(v.record.duration for v in passed)
            rejected_dur = sum(v.record.duration for v in rejected)

            logger.info(
                f"  Pass  : {len(passed):4d} slices ({format_duration(passed_dur)})"
            )
            logger.info(
                f"  Reject: {len(rejected):4d} slices ({format_duration(rejected_dur)})"
            )

            if passed:
                sims = [v.similarity for v in passed]
                logger.info(
                    f"  Similarity (passed): "
                    f"min={min(sims):.3f}  avg={np.mean(sims):.3f}  max={max(sims):.3f}"
                )

            if not passed:
                logger.warning(
                    f"  ⚠ Không có slice nào pass! "
                    f"Thử giảm similarity_threshold "
                    f"(hiện tại: {step_cfg['similarity_threshold']})"
                )
                logger.warning("  Gợi ý: chạy --inspect để xem phân phối score")
                results["failed"].append(src_name)
                continue

            # ── 4d: Copy sang step04_filtered/ ───────────────────────────────
            verified_updated = copy_passed_slices(verified, step_output_dir, logger)
            all_verified.extend(verified_updated)

            elapsed = time.time() - t_start
            logger.info(f"  ✓ Xong {src_name} ({format_duration(elapsed)})")
            results["success"].append(src_name)

    finally:
        engine.release()

    # ── Lưu filtered manifest ─────────────────────────────────────────────────
    manifest_out = step_output_dir / "filtered_manifest.csv"
    save_filtered_manifest(all_verified, manifest_out, logger)

    # ── Tổng kết ──────────────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    passed_all = [v for v in all_verified if v.passed]
    total_dur  = sum(v.record.duration for v in passed_all)

    logger.info("\n" + "=" * 60)
    logger.info("  TỔNG KẾT BƯỚC 4")
    logger.info("=" * 60)
    logger.info(f"  ✓ Thành công : {len(results['success'])} source(s)")
    logger.info(f"  ⏭ Bỏ qua    : {len(results['skipped'])} source(s) (cache)")
    logger.info(f"  ✗ Thất bại  : {len(results['failed'])} source(s)")
    logger.info(f"  Slices giữ lại : {len(passed_all)}")
    logger.info(f"  Tổng thời lượng: {format_duration(total_dur)}")
    logger.info(f"  Thời gian chạy : {format_duration(total_elapsed)}")

    if results["failed"]:
        logger.warning("\nSource thất bại:")
        for s in results["failed"]:
            logger.warning(f"  - {s}")
        logger.warning("\nGợi ý:")
        logger.warning("  - Giảm similarity_threshold: 0.70 → 0.60")
        logger.warning("  - Tăng n_bootstrap_segments: 30 → 50")
        logger.warning("  - Chạy với --inspect để xem phân phối score")

    if total_dur < 1800:
        logger.warning(
            f"\n⚠ Dataset chỉ có {format_duration(total_dur)} "
            f"(< 30 phút) — StyleTTS2 cần tối thiểu 30 phút. "
            f"Hãy thêm audio nguồn."
        )
    else:
        logger.info(f"\n✅ {format_duration(total_dur)} — Đủ điều kiện!")

    logger.info("\n→ Chạy tiếp: python step05_transcription.py")

if __name__ == "__main__":
    main()