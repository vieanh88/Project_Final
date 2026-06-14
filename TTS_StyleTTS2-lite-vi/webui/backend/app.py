"""
=============================================================
  FastAPI APP — Demo UI cho StyleTTS2-lite-vi (truyện ma -> audiobook)
=============================================================
Chạy (từ thư mục TTS_StyleTTS2-lite-vi/):
    uvicorn webui.backend.app:app --reload --port 8000
hoặc:
    python webui/backend/app.py

Sau đó mở http://127.0.0.1:8000

Pipeline demo:
  [Audiobook]  text truyện -> (D2 Gemini) script -> sửa bảng -> (D3) audiobook
  [Playground] 1 câu + chọn giọng -> nghe ngay

Endpoints:
  GET  /                     -> frontend
  GET  /api/health           -> engine info (device, checkpoint, params)
  GET  /api/config           -> default params (speed, denoise, model…)
  GET  /api/voices           -> danh sách giọng (builtin + upload)
  POST /api/voices/upload    -> upload .wav giọng (multipart) + health-check
  GET  /api/sample-stories          -> list truyện mẫu
  GET  /api/sample-stories/{name}   -> full text 1 truyện mẫu
  POST /api/generate-script  -> D2: text -> script lines (Gemini)
  POST /api/playground       -> D0: 1 câu -> audio/wav (stream)
  POST /api/synthesize       -> D3: tạo job tổng hợp audiobook -> {job_id}
  GET  /api/jobs/{job_id}    -> tiến độ job
  /media/*                   -> file audiobook đã sinh
=============================================================
"""

from __future__ import annotations

import sys

# Ép stdout/stderr sang UTF-8 — tránh UnicodeEncodeError khi print tiếng Việt/emoji
# trên Windows (console/redirect mặc định cp1252). Phải chạy TRƯỚC khi import engine
# vì logger của inference_engine giữ tham chiếu sys.stdout lúc import.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ----- Đường dẫn + import pipeline -----
BACKEND_DIR = Path(__file__).resolve().parent          # webui/backend/
WEBUI_DIR = BACKEND_DIR.parent                          # webui/
PROJECT_ROOT = WEBUI_DIR.parent                         # TTS_StyleTTS2-lite-vi/
FRONTEND_DIR = WEBUI_DIR / "frontend"
INFERENCE_DIR = PROJECT_ROOT / "inference"
for _p in (str(BACKEND_DIR), str(INFERENCE_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from pipeline import TTSPipeline, AUDIO_DIR, ROLES       # noqa: E402
from jobs import JobManager                              # noqa: E402
from config_loader import cfg_value                      # noqa: E402

pipeline = TTSPipeline()
jobs = JobManager()


# ============================================================
# Lifespan — load model 1 lần lúc startup
# ============================================================
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=" * 60)
    print("  Loading TTS engine (1 lần)… vui lòng đợi.")
    print("=" * 60)
    await run_in_threadpool(pipeline.load)
    print("  ✅ Engine sẵn sàng. Mở http://127.0.0.1:8000")
    yield
    print("  Shutting down.")


app = FastAPI(title="AI Story Narrator Demo", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


# ============================================================
# Request schemas
# ============================================================
class GenScriptReq(BaseModel):
    text: str
    model: Optional[str] = None
    thinking: Optional[bool] = None
    chunk_size: Optional[int] = None
    max_retries: Optional[int] = None
    dry_run: bool = False


class PlaygroundReq(BaseModel):
    text: str
    voice_id: str
    speed: float = 1.0
    denoise: float = 0.3
    split_dur: float = 2.0


class ScriptLineIn(BaseModel):
    id: int = 1
    role: str = "narrator"
    text: str = ""
    pause_after_ms: int = 500


class SynthReq(BaseModel):
    script: List[ScriptLineIn]
    male_voice_id: str = "builtin_male"
    female_voice_id: str = "builtin_female"
    narrator_speed: float = 1.0
    character_speed: float = 1.0
    denoise: float = 0.3
    split_dur: float = 2.0
    skip_on_error: bool = True
    normalize: bool = True


# ============================================================
# API — info / config / voices
# ============================================================
@app.get("/api/health")
async def api_health():
    return pipeline.info()


@app.get("/api/config")
async def api_config():
    c = pipeline.cfg
    return {
        "roles": list(ROLES),
        "tts": {
            "narrator_speed": cfg_value(c, "tts", "narrator_speed"),
            "character_speed": cfg_value(c, "tts", "character_speed"),
            "denoise": cfg_value(c, "tts", "denoise"),
            "split_dur": cfg_value(c, "tts", "split_dur"),
            "skip_on_error": cfg_value(c, "tts", "skip_on_error"),
            "normalize": cfg_value(c, "tts", "normalize"),
        },
        "nlp": {
            "model": cfg_value(c, "nlp", "model"),
            "thinking": cfg_value(c, "nlp", "thinking"),
        },
    }


@app.get("/api/voices")
async def api_voices():
    return {"voices": pipeline.public_voices()}


@app.post("/api/voices/upload")
async def api_upload_voice(file: UploadFile = File(...), role: str = Form("male")):
    raw = await file.read()
    if not raw:
        raise HTTPException(400, "File rỗng.")
    try:
        result = await run_in_threadpool(
            pipeline.add_uploaded_voice, raw, file.filename or "voice.wav", role
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    return result


# ============================================================
# API — sample stories
# ============================================================
def _sample_dir() -> Path:
    return PROJECT_ROOT / "data" / "raw_stories"


@app.get("/api/sample-stories")
async def api_sample_stories():
    import nlp_generator as nlp
    out = []
    d = _sample_dir()
    if d.exists():
        for p in sorted(d.glob("*.txt")):
            try:
                text = nlp.read_input_text(p)
            except Exception:
                continue
            out.append({
                "name": p.stem,
                "chars": len(text),
                "preview": text.strip()[:140],
            })
    return {"stories": out}


@app.get("/api/sample-stories/{name}")
async def api_sample_story(name: str):
    import nlp_generator as nlp
    p = _sample_dir() / f"{name}.txt"
    if not p.exists():
        raise HTTPException(404, "Không tìm thấy truyện mẫu.")
    return {"name": name, "text": nlp.read_input_text(p)}


# ============================================================
# API — D2 generate script
# ============================================================
@app.post("/api/generate-script")
async def api_generate_script(req: GenScriptReq):
    try:
        result = await run_in_threadpool(
            pipeline.generate_script,
            req.text, req.model, req.thinking,
            req.chunk_size, req.max_retries, req.dry_run,
        )
    except (ValueError, RuntimeError, ImportError) as e:
        raise HTTPException(400, str(e))
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return result


# ============================================================
# API — Playground (1 câu)
# ============================================================
@app.post("/api/playground")
async def api_playground(req: PlaygroundReq):
    try:
        wav_bytes = await run_in_threadpool(
            pipeline.synth_sentence,
            req.text, req.voice_id, req.speed, req.denoise, req.split_dur,
        )
    except KeyError as e:
        raise HTTPException(404, str(e))
    except ValueError as e:
        raise HTTPException(400, str(e))
    except Exception as e:                               # noqa: BLE001
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    return Response(content=wav_bytes, media_type="audio/wav")


# ============================================================
# API — D3 synthesize audiobook (background job)
# ============================================================
@app.post("/api/synthesize")
async def api_synthesize(req: SynthReq):
    if not req.script:
        raise HTTPException(400, "Script rỗng.")
    for vid in (req.male_voice_id, req.female_voice_id):
        if vid not in pipeline.voices:
            raise HTTPException(400, f"Voice id không tồn tại: {vid}")

    script = [ln.model_dump() for ln in req.script]

    def task(progress_cb):
        res = pipeline.synth_audiobook(
            script=script,
            male_voice_id=req.male_voice_id,
            female_voice_id=req.female_voice_id,
            narrator_speed=req.narrator_speed,
            character_speed=req.character_speed,
            denoise=req.denoise,
            split_dur=req.split_dur,
            skip_on_error=req.skip_on_error,
            normalize=req.normalize,
            progress_cb=progress_cb,
        )
        return {
            "audio_url": f"/media/{res['out_path'].name}",
            "summary": res["summary"],
            "stats": res["stats"],
        }

    job_id = jobs.submit(task, label="audiobook")
    return {"job_id": job_id}


@app.get("/api/jobs/{job_id}")
async def api_job(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404, "Job không tồn tại.")
    return job


# ============================================================
# Static mounts (đăng ký SAU API routes)
# ============================================================
app.mount("/media", StaticFiles(directory=str(AUDIO_DIR)), name="media")
app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
