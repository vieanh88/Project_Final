"""
=============================================================
  PIPELINE MODULES - Xử lý audio để tách giọng Nguyễn Ngọc Ngạn
=============================================================
Các module chính:
  1. VocalSeparator   — Demucs: tách nhạc nền
  2. SpeakerDiarizer  — pyannote: phân đoạn người nói
  3. SpeakerVerifier  — Embedding: xác định giọng Ngạn
  4. QualityFilter    — Lọc đoạn xấu (hét, noise, clipping)
  5. AudioNormalizer  — Chuẩn hóa output theo chuẩn XTTS v2
=============================================================
"""

import os
import gc
import csv
import logging
import warnings
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from tqdm import tqdm

import sys

# Tắt warning không cần thiết
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger("speechbrain").setLevel(logging.WARNING)


# ============================================================
#  DATA CLASSES
# ============================================================

@dataclass
class Segment:
    """Đại diện một đoạn audio được phân đoạn."""
    speaker_id: str         # "SPEAKER_00", "SPEAKER_01", ...
    start: float            # giây
    end: float              # giây
    audio_path: str         # path đến file vocals.wav gốc
    is_ngan: bool = False   # sau speaker verification
    similarity_score: float = 0.0
    snr_db: float = 0.0
    passed_quality: bool = False
    output_path: str = ""

    @property
    def duration(self) -> float:
        return self.end - self.start


@dataclass
class ProcessingResult:
    """Kết quả xử lý 1 file audio."""
    input_file: str
    vocals_file: str = ""
    total_segments: int = 0
    ngan_segments: int = 0
    passed_segments: int = 0
    failed_reason_counts: Dict[str, int] = field(default_factory=dict)
    segments: List[Segment] = field(default_factory=list)


# ============================================================
#  MODULE 1: VOCAL SEPARATION
# ============================================================

class VocalSeparator:
    """
    Tách nhạc nền ra khỏi giọng nói bằng Demucs.
    Output: file vocals.wav trong thư mục workdir
    """

    def __init__(self, model: str = "htdemucs_ft", cpu_only: bool = False):
        self.model = model
        self.device = "cpu" if cpu_only else ("cuda" if torch.cuda.is_available() else "cpu")
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.info(f"VocalSeparator init — model={model}, device={self.device}")

    def separate(self, audio_path: str, output_dir: str) -> str:
        """
        Tách vocals từ file audio.

        Args:
            audio_path: Đường dẫn file audio gốc
            output_dir:  Thư mục lưu kết quả

        Returns:
            Đường dẫn đến file vocals.wav
        """
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Kiểm tra cache — nếu đã tách rồi thì skip
        stem_name = audio_path.stem
        expected_vocals = output_dir / self.model / stem_name / "vocals.wav"

        if expected_vocals.exists():
            self.logger.info(f"Cache hit — vocals đã tồn tại: {expected_vocals}")
            return str(expected_vocals)

        self.logger.info(f"Đang tách vocals: {audio_path.name} ...")

        # Ép tiến trình dùng UTF-8 để tránh lỗi encoding trên Windows
        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"

        # Gọi demucs qua command line để tránh memory leak khi xử lý nhiều file
        cmd = [
            sys.executable, "-m", "demucs",
            "--two-stems=vocals",      # Chỉ cần vocals + accompaniment
            "-n", self.model,
            "-d", self.device,
            "--out", str(output_dir),
            str(audio_path),
        ]

        result = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8")

        if result.returncode != 0:
            raise RuntimeError(
                f"Demucs thất bại cho {audio_path.name}:\n{result.stderr}"
            )

        if not expected_vocals.exists():
            # Demucs có thể tạo cấu trúc thư mục khác nhau
            # Tìm file vocals.wav trong output
            vocals_files = list(output_dir.rglob("vocals.wav"))
            if not vocals_files:
                raise FileNotFoundError(f"Không tìm thấy vocals.wav sau khi chạy Demucs cho {audio_path.name}")
            expected_vocals = vocals_files[0]

        self.logger.info(f"Tách xong → {expected_vocals}")
        return str(expected_vocals)

    def release(self):
        """Giải phóng VRAM sau khi dùng xong."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


# ============================================================
#  MODULE 2: SPEAKER DIARIZATION
# ============================================================

class SpeakerDiarizer:
    """
    Phân đoạn người nói bằng pyannote-audio 3.x.
    Yêu cầu: HF token + accept terms tại HuggingFace.
    """

    def __init__(self, hf_token: str, model_name: str = "pyannote/speaker-diarization-3.1",
                 max_speakers: int = 6, merge_gap: float = 0.5):
        self.hf_token = hf_token
        self.model_name = model_name
        self.max_speakers = max_speakers
        self.merge_gap = merge_gap
        self.logger = logging.getLogger(self.__class__.__name__)
        self._pipeline = None

    def _load_pipeline(self):
        """Lazy load — chỉ load khi cần để tiết kiệm VRAM."""
        if self._pipeline is None:
            self.logger.info("Đang load pyannote diarization pipeline...")
            from pyannote.audio import Pipeline
            self._pipeline = Pipeline.from_pretrained(
                self.model_name,
                use_auth_token=self.hf_token
            )
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self._pipeline.to(device)
            self.logger.info(f"Pyannote pipeline loaded trên {device}")

    def diarize(self, vocals_path: str, rttm_output_path: Optional[str] = None) -> List[Segment]:
        """
        Chạy diarization trên file vocals.wav.

        Args:
            vocals_path:       Đường dẫn file vocals.wav (sau Demucs)
            rttm_output_path:  Nếu không None, lưu file RTTM để debug

        Returns:
            Danh sách các Segment đã được merge
        """
        self._load_pipeline()
        vocals_path = Path(vocals_path)
        self.logger.info(f"Diarization: {vocals_path.name} (max_speakers={self.max_speakers})")

        diarization = self._pipeline(
            str(vocals_path),
            max_speakers=self.max_speakers
        )

        # Lưu RTTM nếu cần
        if rttm_output_path:
            Path(rttm_output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(rttm_output_path, "w") as f:
                diarization.write_rttm(f)
            self.logger.info(f"Đã lưu RTTM → {rttm_output_path}")

        # Chuyển diarization sang danh sách Segment
        raw_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            raw_segments.append(Segment(
                speaker_id=speaker,
                start=turn.start,
                end=turn.end,
                audio_path=str(vocals_path)
            ))

        # Sắp xếp theo thời gian
        raw_segments.sort(key=lambda s: s.start)

        # Merge các segment gần nhau của cùng speaker
        merged = self._merge_segments(raw_segments)

        self.logger.info(
            f"Diarization xong: {len(raw_segments)} raw → {len(merged)} segments sau merge"
        )
        return merged

    def _merge_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Gộp các segment liên tiếp của cùng speaker nếu khoảng cách < merge_gap.
        """
        if not segments:
            return []

        merged = [segments[0]]
        for seg in segments[1:]:
            last = merged[-1]
            # Cùng speaker VÀ khoảng cách nhỏ hơn ngưỡng
            if (seg.speaker_id == last.speaker_id and
                    seg.start - last.end <= self.merge_gap):
                # Mở rộng segment cuối
                last.end = seg.end
            else:
                merged.append(seg)
        return merged

    def get_dominant_speaker(self, segments: List[Segment]) -> str:
        """
        Xác định speaker có tổng thời lượng lớn nhất.
        Đây chính là giọng Nguyễn Ngọc Ngạn (narrator chiếm đa số).
        """
        duration_map: Dict[str, float] = {}
        for seg in segments:
            duration_map[seg.speaker_id] = duration_map.get(seg.speaker_id, 0) + seg.duration

        dominant = max(duration_map, key=duration_map.get)
        total = sum(duration_map.values())

        self.logger.info("Thống kê thời lượng theo speaker:")
        for spk, dur in sorted(duration_map.items(), key=lambda x: -x[1]):
            self.logger.info(f"  {spk}: {dur:.1f}s ({100*dur/total:.1f}%)")

        dominant_pct = 100 * duration_map[dominant] / total
        self.logger.info(
            f"→ Dominant speaker (giọng Ngạn): {dominant} "
            f"({duration_map[dominant]:.1f}s, {dominant_pct:.1f}%)"
        )

        if dominant_pct < 40.0:
            self.logger.warning(
                f"Dominant speaker chỉ chiếm {dominant_pct:.1f}% — "
                f"thấp hơn mong đợi (>40%). Kiểm tra lại file audio."
            )
        return dominant

    def release(self):
        """Giải phóng VRAM sau khi diarization xong."""
        self._pipeline = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Đã giải phóng VRAM sau diarization.")


# ============================================================
#  MODULE 3: SPEAKER VERIFICATION
# ============================================================

class SpeakerVerifier:
    """
    Xác minh từng segment có phải giọng Ngạn không
    bằng cách so sánh speaker embedding (cosine similarity).

    Sử dụng pyannote/embedding (ECAPA-TDNN).
    """

    def __init__(self, hf_token: str, similarity_threshold: float = 0.75,
                 n_reference: int = 20, min_ref_duration: float = 5.0):
        self.hf_token = hf_token
        self.threshold = similarity_threshold
        self.n_reference = n_reference
        self.min_ref_duration = min_ref_duration
        self.logger = logging.getLogger(self.__class__.__name__)
        self._model = None
        self._reference_embedding: Optional[np.ndarray] = None

    def _load_model(self):
        if self._model is None:
            self.logger.info("Đang load speaker embedding model...")
            from pyannote.audio import Inference
            self._model = Inference(
                "pyannote/embedding",
                window="whole",
                use_auth_token=self.hf_token
            )
            self.logger.info("Speaker embedding model loaded.")

    def build_reference_from_dominant(
        self,
        segments: List[Segment],
        dominant_speaker_id: str
    ) -> np.ndarray:
        """
        Xây dựng reference embedding của giọng Ngạn từ các segment
        của dominant speaker (bootstrap — không cần audio sạch trước).

        Chọn N segment chất lượng cao nhất (duration dài, không clipping).
        """
        self._load_model()

        # Lọc segment của dominant speaker có đủ độ dài
        candidates = [
            s for s in segments
            if s.speaker_id == dominant_speaker_id
            and s.duration >= self.min_ref_duration
        ]

        # Ưu tiên segment dài nhất (thường ít bị overlap nhạc nền nhất)
        candidates.sort(key=lambda s: s.duration, reverse=True)
        selected = candidates[:self.n_reference]

        if len(selected) < 3:
            raise ValueError(
                f"Không đủ segment tham chiếu (tìm được {len(selected)}, cần ít nhất 3). "
                f"Thử giảm MIN_REFERENCE_DURATION."
            )

        self.logger.info(
            f"Xây dựng reference embedding từ {len(selected)} segments "
            f"(tổng {sum(s.duration for s in selected):.1f}s)"
        )

        # Tính embedding cho từng segment rồi lấy trung bình (centroid)
        embeddings = []
        for seg in tqdm(selected, desc="Building reference embeddings"):
            emb = self._get_embedding_for_segment(seg)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            raise ValueError("Không tính được embedding nào. Kiểm tra file audio.")

        reference_emb = np.mean(embeddings, axis=0)
        reference_emb /= np.linalg.norm(reference_emb)  # normalize
        self._reference_embedding = reference_emb

        self.logger.info(
            f"Reference embedding đã xây xong từ {len(embeddings)} segments."
        )
        return reference_emb

    def verify_segments(self, segments: List[Segment]) -> List[Segment]:
        """
        Tính cosine similarity giữa từng segment và reference embedding.
        Gán is_ngan=True nếu vượt ngưỡng.
        """
        if self._reference_embedding is None:
            raise RuntimeError("Cần gọi build_reference_from_dominant() trước.")

        self._load_model()
        self.logger.info(f"Verifying {len(segments)} segments (threshold={self.threshold})")

        ngan_count = 0
        for seg in tqdm(segments, desc="Speaker verification"):
            emb = self._get_embedding_for_segment(seg)
            if emb is not None:
                similarity = float(np.dot(emb, self._reference_embedding))
                seg.similarity_score = similarity
                seg.is_ngan = similarity >= self.threshold
                if seg.is_ngan:
                    ngan_count += 1
            else:
                seg.is_ngan = False
                seg.similarity_score = 0.0

        self.logger.info(
            f"Verification xong: {ngan_count}/{len(segments)} segments xác nhận là giọng Ngạn."
        )
        return segments

    def _get_embedding_for_segment(self, seg: Segment) -> Optional[np.ndarray]:
        """Trích xuất embedding cho 1 segment."""
        try:
            from pyannote.audio import Inference
            from pyannote.core import Segment as PyannoteSegment

            audio, sr = torchaudio.load(seg.audio_path)

            # Cắt đoạn audio theo [start, end]
            start_sample = int(seg.start * sr)
            end_sample = int(seg.end * sr)
            audio_chunk = audio[:, start_sample:end_sample]

            # Quá ngắn → skip
            if audio_chunk.shape[1] < sr * 1.0:
                return None

            # Chạy embedding inference
            waveform = {"waveform": audio_chunk, "sample_rate": sr}
            embedding = self._model(waveform)

            if isinstance(embedding, np.ndarray):
                emb = embedding.flatten()
            else:
                emb = embedding.numpy().flatten()

            # Normalize L2
            norm = np.linalg.norm(emb)
            if norm < 1e-8:
                return None
            return emb / norm

        except Exception as e:
            self.logger.debug(f"Lỗi khi tính embedding cho segment {seg.start:.1f}-{seg.end:.1f}: {e}")
            return None

    def release(self):
        self._model = None
        self._reference_embedding = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Đã giải phóng VRAM sau speaker verification.")


# ============================================================
#  MODULE 4: QUALITY FILTER
# ============================================================

class QualityFilter:
    """
    Lọc các segment kém chất lượng:
    - Thời lượng không phù hợp (< 6s hoặc > 30s)
    - SNR quá thấp (còn nhiều nhạc nền)
    - Tiếng hét / âm thanh méo (clipping)
    - Tỉ lệ im lặng quá cao
    """

    def __init__(
        self,
        min_duration: float = 6.0,
        max_duration: float = 30.0,
        min_snr_db: float = 20.0,
        scream_ratio: float = 4.0,
        max_clipping_pct: float = 0.5,
        max_silence_ratio: float = 0.4,
    ):
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.min_snr_db = min_snr_db
        self.scream_ratio = scream_ratio
        self.max_clipping_pct = max_clipping_pct
        self.max_silence_ratio = max_silence_ratio
        self.logger = logging.getLogger(self.__class__.__name__)

    def filter(self, segments: List[Segment]) -> Tuple[List[Segment], Dict[str, int]]:
        """
        Lọc danh sách segment, chỉ giữ lại những đoạn chất lượng tốt.

        Returns:
            (passed_segments, failed_reason_counts)
        """
        # Chỉ xét segment đã xác nhận là giọng Ngạn
        ngan_segments = [s for s in segments if s.is_ngan]
        self.logger.info(f"Quality filter: kiểm tra {len(ngan_segments)} segments giọng Ngạn")

        passed = []
        failed_reasons: Dict[str, int] = {}

        for seg in tqdm(ngan_segments, desc="Quality filtering"):
            reason = self._check_segment(seg)
            if reason is None:
                seg.passed_quality = True
                passed.append(seg)
            else:
                failed_reasons[reason] = failed_reasons.get(reason, 0) + 1
                self.logger.debug(
                    f"Loại bỏ [{seg.start:.1f}-{seg.end:.1f}s]: {reason}"
                )

        self.logger.info(
            f"Filter xong: {len(passed)}/{len(ngan_segments)} segments pass. "
            f"Lý do loại: {failed_reasons}"
        )
        return passed, failed_reasons

    def _check_segment(self, seg: Segment) -> Optional[str]:
        """
        Kiểm tra 1 segment.
        Returns: None nếu pass, hoặc chuỗi mô tả lý do bị loại.
        """
        # 1. Kiểm tra thời lượng
        if seg.duration < self.min_duration:
            return f"too_short({seg.duration:.1f}s < {self.min_duration}s)"
        if seg.duration > self.max_duration:
            return f"too_long({seg.duration:.1f}s > {self.max_duration}s)"

        # Load audio của segment này
        try:
            audio, sr = librosa.load(
                seg.audio_path,
                sr=None,
                offset=seg.start,
                duration=seg.duration,
                mono=True
            )
        except Exception as e:
            return f"load_error({e})"

        if len(audio) < sr * self.min_duration:
            return "too_short_after_load"

        # 2. Clipping detection — audio bị méo/vỡ
        clipping_pct = self._check_clipping(audio)
        if clipping_pct > self.max_clipping_pct:
            return f"clipping({clipping_pct:.1f}% > {self.max_clipping_pct}%)"

        # 3. Scream detection — tiếng hét gây méo mô hình
        if self._detect_scream(audio, sr):
            return "scream_detected"

        # 4. Silence ratio — đoạn chủ yếu là im lặng
        silence_ratio = self._check_silence_ratio(audio, sr)
        if silence_ratio > self.max_silence_ratio:
            return f"too_much_silence({silence_ratio:.1%} > {self.max_silence_ratio:.1%})"

        # 5. SNR estimate — còn nhiều nhạc nền
        snr = self._estimate_snr(audio, sr)
        seg.snr_db = snr
        if snr < self.min_snr_db:
            return f"low_snr({snr:.1f}dB < {self.min_snr_db}dB)"

        return None  # Pass tất cả

    def _check_clipping(self, audio: np.ndarray) -> float:
        """% mẫu audio bị clipping (|amplitude| > 0.99)."""
        clipped = np.sum(np.abs(audio) > 0.99)
        return 100.0 * clipped / len(audio)

    def _detect_scream(self, audio: np.ndarray, sr: int) -> bool:
        """
        Phát hiện tiếng hét dựa trên:
        - Peak RMS / mean RMS ratio cao bất thường
        - Spectral flatness cao (âm thanh vô tổ chức)
        """
        # Tính RMS theo frame
        frame_length = int(sr * 0.025)   # 25ms frame
        hop_length   = int(sr * 0.010)   # 10ms hop
        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        if len(rms) == 0 or np.mean(rms) < 1e-8:
            return False

        # Ratio peak/mean — tiếng hét thường rất đột ngột và mạnh
        peak_ratio = np.max(rms) / (np.mean(rms) + 1e-8)
        if peak_ratio > self.scream_ratio:
            return True

        # Spectral flatness — giọng nói bình thường có flatness thấp
        # Tiếng hét / âm thanh méo có flatness cao hơn
        flatness = librosa.feature.spectral_flatness(y=audio, hop_length=hop_length)[0]
        if np.mean(flatness) > 0.3:
            return True

        return False

    def _check_silence_ratio(self, audio: np.ndarray, sr: int) -> float:
        """Tỉ lệ frame im lặng (RMS < ngưỡng)."""
        frame_length = int(sr * 0.025)
        hop_length   = int(sr * 0.010)
        rms = librosa.feature.rms(
            y=audio, frame_length=frame_length, hop_length=hop_length
        )[0]

        # Ngưỡng im lặng: -40dB so với max RMS
        max_rms = np.max(rms)
        silence_threshold = max_rms * 0.01  # -40dB
        silence_frames = np.sum(rms < silence_threshold)
        return silence_frames / len(rms)

    def _estimate_snr(self, audio: np.ndarray, sr: int) -> float:
        """
        Ước lượng SNR đơn giản dựa trên phổ âm thanh.
        Phương pháp: so sánh năng lượng trong dải giọng nói (300-3400Hz)
        với dải noise (>6000Hz hoặc <80Hz).

        Đây là ước lượng gần đúng — đủ để lọc các đoạn còn nhiều nhạc nền.
        """
        # Tính power spectrum
        n_fft = 2048
        stft = np.abs(librosa.stft(audio, n_fft=n_fft))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        # Dải giọng nói
        voice_mask = (freqs >= 300) & (freqs <= 3400)
        # Dải noise (tần số thấp và cao — thường là nhạc nền sau Demucs)
        noise_mask = (freqs < 80) | (freqs > 6000)

        voice_energy = np.mean(stft[voice_mask, :] ** 2)
        noise_energy = np.mean(stft[noise_mask, :] ** 2)

        if noise_energy < 1e-10:
            return 60.0  # Rất sạch

        snr = 10 * np.log10(voice_energy / noise_energy + 1e-10)
        return float(snr)


# ============================================================
#  MODULE 5: AUDIO NORMALIZER & EXPORTER
# ============================================================

class AudioNormalizer:
    """
    Chuẩn hóa và xuất các segment đã pass thành file WAV
    theo chuẩn XTTS v2:
    - 22050 Hz, mono, 16-bit WAV
    - Normalized LUFS
    - Padding im lặng ở đầu/cuối
    """

    def __init__(
        self,
        output_dir: str,
        sample_rate: int = 22050,
        target_lufs: float = -20.0,
        silence_pad_ms: int = 100,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = sample_rate
        self.target_lufs = target_lufs
        self.silence_pad_ms = silence_pad_ms
        self.logger = logging.getLogger(self.__class__.__name__)

    def export_segments(
        self,
        segments: List[Segment],
        source_name: str
    ) -> List[str]:
        """
        Xuất các segment đã pass thành file WAV.

        Args:
            segments:    Danh sách segment đã pass quality filter
            source_name: Tên file gốc (dùng để đặt tên output)

        Returns:
            Danh sách đường dẫn file đã xuất
        """
        # Tạo sub-folder theo tên file gốc
        out_subdir = self.output_dir / source_name
        out_subdir.mkdir(parents=True, exist_ok=True)

        exported_paths = []
        pad_samples = int(self.silence_pad_ms * self.sample_rate / 1000)
        silence_pad = np.zeros(pad_samples, dtype=np.float32)

        for i, seg in enumerate(tqdm(segments, desc=f"Exporting {source_name}")):
            try:
                # Load audio của segment
                audio, sr = librosa.load(
                    seg.audio_path,
                    sr=self.sample_rate,  # Resample ngay khi load
                    offset=seg.start,
                    duration=seg.duration,
                    mono=True
                )

                # Normalize âm lượng
                audio = self._normalize_loudness(audio)

                # Thêm padding im lặng ở đầu và cuối
                audio = np.concatenate([silence_pad, audio, silence_pad])

                # Đặt tên file output
                out_filename = f"{source_name}_seg{i:04d}_{seg.start:.1f}-{seg.end:.1f}s.wav"
                out_path = out_subdir / out_filename

                # Lưu file WAV 16-bit
                sf.write(str(out_path), audio, self.sample_rate, subtype="PCM_16")

                seg.output_path = str(out_path)
                exported_paths.append(str(out_path))

            except Exception as e:
                self.logger.warning(f"Lỗi khi export segment {i}: {e}")
                continue

        self.logger.info(f"Đã xuất {len(exported_paths)} files → {out_subdir}")
        return exported_paths

    def _normalize_loudness(self, audio: np.ndarray) -> np.ndarray:
        """
        Normalize âm lượng về target LUFS gần đúng.
        Dùng RMS normalization thay vì pyloudnorm để tránh dependency thêm.
        Target: -20 LUFS ≈ RMS = 0.1 (-20dBFS)
        """
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 1e-8:
            return audio  # Im lặng hoàn toàn, không normalize

        # -20 LUFS ≈ RMS target = 0.1 (tương đương -20dBFS)
        target_rms = 10 ** (self.target_lufs / 20.0)
        gain = target_rms / rms
        audio = audio * gain

        # Clip để tránh overflow
        audio = np.clip(audio, -1.0, 1.0)
        return audio

    def save_metadata_csv(self, all_segments: List[Segment], output_csv: str):
        """
        Lưu file CSV metadata cho toàn bộ dataset.
        Format: path|text (tương thích với XTTS v2 training)
        """
        csv_path = Path(output_csv)
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f, delimiter="|")
            writer.writerow([
                "audio_file", "speaker_id", "source_file",
                "start_s", "end_s", "duration_s",
                "similarity_score", "snr_db", "passed_quality"
            ])
            for seg in all_segments:
                writer.writerow([
                    seg.output_path,
                    seg.speaker_id,
                    seg.audio_path,
                    f"{seg.start:.3f}",
                    f"{seg.end:.3f}",
                    f"{seg.duration:.3f}",
                    f"{seg.similarity_score:.4f}",
                    f"{seg.snr_db:.2f}",
                    seg.passed_quality
                ])
        self.logger.info(f"Đã lưu metadata → {csv_path}")