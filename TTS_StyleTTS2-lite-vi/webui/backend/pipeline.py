"""
=============================================================
  WEBUI PIPELINE — Cầu nối FastAPI <-> code inference có sẵn
=============================================================
Tái sử dụng TRỰC TIẾP các module trong inference/ (không gọi CLI):
  - config_loader        : đọc inference_config.yaml (đường dẫn engine/refs)
  - inference_engine     : StyleTTS2LiteVNInference (load model 1 lần)
  - tts_generator        : normalize_audio (+ logic ghép audiobook mô phỏng D3)
  - nlp_generator        : Gemini text -> script.json (D2)
  - download_female_ref  : health_check cho file giọng upload (D1)

Thiết kế:
  - Engine load 1 LẦN lúc startup, giữ trong VRAM (3050Ti 4GB, FP16).
  - 1 threading.Lock bao mọi lời gọi GPU (engine KHÔNG thread-safe) ->
    background job tổng hợp audiobook + request playground tự serialize.
  - Cache style vector theo (ref_path, denoise, split_dur) để khỏi tính lại.
  - Voice registry: 2 giọng builtin (từ config) + giọng user upload.
=============================================================
"""

from __future__ import annotations

import gc
import io
import sys
import threading
import uuid
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import soundfile as sf
import torch

# ----- Định vị thư mục + nạp code inference/ vào sys.path -----
BACKEND_DIR = Path(__file__).resolve().parent          # webui/backend/
WEBUI_DIR = BACKEND_DIR.parent                          # webui/
PROJECT_ROOT = WEBUI_DIR.parent                         # TTS_StyleTTS2-lite-vi/
INFERENCE_DIR = PROJECT_ROOT / "inference"
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from config_loader import (                              # noqa: E402
    DEFAULT_CONFIG_PATH, load_config, cfg_value, resolve_path,
)
from inference_engine import StyleTTS2LiteVNInference, SR  # noqa: E402
from tts_generator import normalize_audio                 # noqa: E402
from download_female_ref import health_check              # noqa: E402

# Runtime dirs (gitignore): file upload + audio sinh ra
RUNTIME_DIR = BACKEND_DIR / "runtime"
UPLOAD_DIR = RUNTIME_DIR / "uploads"
AUDIO_DIR = RUNTIME_DIR / "audio"
for _d in (UPLOAD_DIR, AUDIO_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ROLES = ("narrator", "character_male", "character_female")


class TTSPipeline:
    """Quản lý engine + style cache + voice registry. Khởi tạo 1 instance toàn app."""

    def __init__(self) -> None:
        self.cfg: dict = {}
        self.engine: Optional[StyleTTS2LiteVNInference] = None
        self._style_cache: dict[tuple, torch.Tensor] = {}
        self._lock = threading.Lock()          # serialize mọi lời gọi GPU
        self.voices: dict[str, dict] = {}       # voice_id -> {id,label,kind,role,path}
        self.ready = False

    # --------------------------------------------------------
    # Startup
    # --------------------------------------------------------
    def load(self) -> None:
        """Load config + engine (nặng, gọi 1 lần lúc startup). Đăng ký giọng builtin."""
        self.cfg = load_config(DEFAULT_CONFIG_PATH)

        self.engine = StyleTTS2LiteVNInference(
            checkpoint_path=resolve_path(cfg_value(self.cfg, "engine", "checkpoint")),
            repo_root=resolve_path(cfg_value(self.cfg, "engine", "repo")),
            config_path=resolve_path(cfg_value(self.cfg, "engine", "model_config")),
            device=cfg_value(self.cfg, "engine", "device"),
            use_fp16=cfg_value(self.cfg, "engine", "use_fp16"),
        )

        # Giọng builtin từ config
        male = resolve_path(cfg_value(self.cfg, "references", "male_ref"))
        female = resolve_path(cfg_value(self.cfg, "references", "female_ref"))
        if male and male.exists():
            self.voices["builtin_male"] = {
                "id": "builtin_male", "label": "Giọng Ngạn (nam) — mặc định",
                "kind": "builtin", "role": "male", "path": male,
            }
        if female and female.exists():
            self.voices["builtin_female"] = {
                "id": "builtin_female", "label": "Giọng nữ — mặc định",
                "kind": "builtin", "role": "female", "path": female,
            }
        self.ready = True

    # --------------------------------------------------------
    # Info / voices
    # --------------------------------------------------------
    def info(self) -> dict:
        cuda = torch.cuda.is_available()
        data = {
            "ready": self.ready,
            "cuda": cuda,
            "gpu_name": torch.cuda.get_device_name(0) if cuda else None,
        }
        if self.engine is not None:
            data.update(self.engine.info())
        return data

    def public_voices(self) -> list[dict]:
        """Danh sách giọng cho frontend (ẩn đường dẫn tuyệt đối)."""
        out = []
        for v in self.voices.values():
            out.append({
                "id": v["id"], "label": v["label"],
                "kind": v["kind"], "role": v["role"],
                "filename": v.get("filename"),
                "duration": v.get("duration"),
            })
        # builtin lên đầu, rồi upload mới nhất
        out.sort(key=lambda x: (x["kind"] != "builtin", x["label"]))
        return out

    def _voice_path(self, voice_id: str) -> Path:
        v = self.voices.get(voice_id)
        if v is None:
            raise KeyError(f"Voice id không tồn tại: {voice_id}")
        return v["path"]

    def add_uploaded_voice(self, raw: bytes, filename: str, role: str) -> dict:
        """Lưu file upload, health-check. Trả về voice dict + report (KHÔNG raise nếu warning)."""
        if role not in ("male", "female"):
            role = "male"
        vid = f"upload_{uuid.uuid4().hex[:10]}"
        safe_name = Path(filename).name or f"{vid}.wav"
        dest = UPLOAD_DIR / f"{vid}_{safe_name}"
        dest.write_bytes(raw)

        report = health_check(dest)
        if not report.get("ok"):
            dest.unlink(missing_ok=True)
            raise ValueError(
                "File giọng không hợp lệ: " + "; ".join(report.get("errors", ["unknown"]))
            )

        label = f"{'Nam' if role == 'male' else 'Nữ'} (upload) — {safe_name}"
        self.voices[vid] = {
            "id": vid, "label": label, "kind": "upload", "role": role,
            "path": dest, "filename": safe_name,
            "duration": report.get("duration_sec"),
        }
        return {
            "voice": {
                "id": vid, "label": label, "kind": "upload", "role": role,
                "filename": safe_name, "duration": report.get("duration_sec"),
            },
            "health": report,
        }

    # --------------------------------------------------------
    # Style cache
    # --------------------------------------------------------
    def get_style(self, ref_path: Path, denoise: float, split_dur: float) -> torch.Tensor:
        key = (str(ref_path), round(float(denoise), 3), round(float(split_dur), 3))
        with self._lock:
            cached = self._style_cache.get(key)
            if cached is None:
                cached = self.engine.compute_style(
                    str(ref_path), denoise=denoise, split_dur=split_dur
                )
                self._style_cache[key] = cached
            return cached

    # --------------------------------------------------------
    # Playground — 1 câu
    # --------------------------------------------------------
    def synth_sentence(
        self, text: str, voice_id: str, speed: float = 1.0,
        denoise: float = 0.3, split_dur: float = 2.0,
    ) -> bytes:
        text = (text or "").strip()
        if not text:
            raise ValueError("Text rỗng.")
        ref = self._voice_path(voice_id)
        style = self.get_style(ref, denoise, split_dur)
        with self._lock:
            wav = self.engine.synthesize(text, style, speed=float(speed))
        return _wav_to_bytes(wav, SR)

    # --------------------------------------------------------
    # Audiobook — loop theo script (mô phỏng đúng logic D3) + progress
    # --------------------------------------------------------
    def synth_audiobook(
        self,
        script: list[dict],
        male_voice_id: str,
        female_voice_id: str,
        narrator_speed: float = 1.0,
        character_speed: float = 1.0,
        denoise: float = 0.3,
        split_dur: float = 2.0,
        skip_on_error: bool = True,
        normalize: bool = True,
        progress_cb: Optional[Callable[[int, int, dict], None]] = None,
        out_path: Optional[Path] = None,
    ) -> dict:
        male_ref = self._voice_path(male_voice_id)
        female_ref = self._voice_path(female_voice_id)

        if progress_cb:
            progress_cb(0, len(script), {"stage": "computing_styles"})
        male_style = self.get_style(male_ref, denoise, split_dur)
        female_style = self.get_style(female_ref, denoise, split_dur)
        styles = {
            "narrator": male_style,
            "character_male": male_style,
            "character_female": female_style,
        }
        speed_per_role = {
            "narrator": float(narrator_speed),
            "character_male": float(character_speed),
            "character_female": float(character_speed),
        }

        chunks: list[np.ndarray] = []
        stats: list[dict] = []
        n = len(script)

        for idx, line in enumerate(script):
            line_id = line.get("id", idx + 1)
            role = line.get("role", "narrator")
            if role not in styles:
                role = "narrator"
            text = (line.get("text") or "").strip()
            pause_ms = int(line.get("pause_after_ms", 500))
            speed = speed_per_role.get(role, 1.0)

            stat = {
                "id": line_id, "role": role,
                "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
                "status": "pending", "speech_dur_sec": 0.0, "error": None,
            }

            if not text:
                stat["status"] = "skipped_empty"
                chunks.append(np.zeros(int(pause_ms / 1000.0 * SR), dtype=np.float32))
                stats.append(stat)
                if progress_cb:
                    progress_cb(idx + 1, n, stat)
                continue

            try:
                with self._lock:
                    wav = self.engine.synthesize(text, styles[role], speed=speed)
                chunks.append(wav.astype(np.float32))
                stat["status"] = "ok"
                stat["speech_dur_sec"] = round(len(wav) / SR, 3)
            except Exception as e:                       # noqa: BLE001
                stat["status"] = "failed"
                stat["error"] = f"{type(e).__name__}: {e}"
                if skip_on_error:
                    chunks.append(np.zeros(int(0.2 * SR), dtype=np.float32))
                else:
                    raise

            if pause_ms > 0:
                chunks.append(np.zeros(int(pause_ms / 1000.0 * SR), dtype=np.float32))

            stats.append(stat)
            if progress_cb:
                progress_cb(idx + 1, n, stat)

            if (idx + 1) % 10 == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        final_wav = (
            np.concatenate(chunks).astype(np.float32) if chunks
            else np.zeros(1, dtype=np.float32)
        )
        if normalize:
            final_wav = normalize_audio(final_wav, target_peak=0.95)

        if out_path is None:
            out_path = AUDIO_DIR / f"audiobook_{uuid.uuid4().hex[:10]}.wav"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), final_wav, SR)

        ok = sum(1 for s in stats if s["status"] == "ok")
        failed = sum(1 for s in stats if s["status"] == "failed")
        skipped = sum(1 for s in stats if s["status"] == "skipped_empty")
        total_audio = len(final_wav) / SR
        role_counts: dict[str, int] = {}
        for s in stats:
            role_counts[s["role"]] = role_counts.get(s["role"], 0) + 1

        return {
            "out_path": out_path,
            "stats": stats,
            "summary": {
                "total_lines": len(stats), "ok_lines": ok,
                "failed_lines": failed, "skipped_empty": skipped,
                "role_counts": role_counts,
                "total_audio_sec": round(total_audio, 2),
                "total_audio_min": round(total_audio / 60, 2),
                "size_mb": round(out_path.stat().st_size / 1e6, 2),
            },
        }

    # --------------------------------------------------------
    # NLP (D2) — text truyện -> script lines qua Gemini
    # --------------------------------------------------------
    def generate_script(
        self, text: str, model: Optional[str] = None,
        thinking: Optional[bool] = None, chunk_size: Optional[int] = None,
        max_retries: Optional[int] = None, dry_run: bool = False,
    ) -> dict:
        import nlp_generator as nlp

        text = (text or "").strip()
        if not text:
            raise ValueError("Text truyện rỗng.")

        model = model or cfg_value(self.cfg, "nlp", "model")
        thinking = cfg_value(self.cfg, "nlp", "thinking") if thinking is None else thinking
        chunk_size = chunk_size or cfg_value(self.cfg, "nlp", "chunk_size")
        max_retries = max_retries or cfg_value(self.cfg, "nlp", "max_retries")
        env_path = resolve_path(cfg_value(self.cfg, "nlp", "env"))

        chunks = nlp.chunk_text_by_paragraph(text, max_chars=int(chunk_size))

        if dry_run:
            return {"dry_run": True, "n_chunks": len(chunks), "lines": []}

        api_key = nlp.load_api_key(env_path)
        client = nlp.build_client(api_key)

        all_lines: list[dict] = []
        current_id = 1
        for chunk in chunks:
            lines = nlp.call_gemini_chunk(
                client, model=model, chunk_text=chunk, start_id=current_id,
                use_thinking=bool(thinking), max_retries=int(max_retries),
            )
            all_lines.extend(lines)
            current_id = len(all_lines) + 1

        all_lines = nlp.post_process_script(all_lines)
        if not all_lines:
            raise RuntimeError("Gemini không sinh được dòng nào. Thử lại / kiểm tra API key.")

        role_counts: dict[str, int] = {}
        for ln in all_lines:
            role_counts[ln["role"]] = role_counts.get(ln["role"], 0) + 1
        est = nlp.estimate_audio_duration_sec(all_lines)

        return {
            "dry_run": False,
            "n_chunks": len(chunks),
            "lines": all_lines,
            "stats": {
                "n_lines": len(all_lines),
                "role_counts": role_counts,
                "estimated_audio_sec": round(est, 1),
                "estimated_audio_min": round(est / 60, 1),
            },
        }


# ============================================================
# Helpers
# ============================================================
def _wav_to_bytes(wav: np.ndarray, sr: int) -> bytes:
    """numpy wav -> WAV bytes (PCM16) để stream về browser."""
    buf = io.BytesIO()
    sf.write(buf, wav, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
