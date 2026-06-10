"""
=============================================================
  JOB MANAGER — Chạy tổng hợp audiobook ở background + theo dõi tiến độ
=============================================================
Tổng hợp audiobook tốn vài phút -> không thể block HTTP request.
  - ThreadPoolExecutor(max_workers=1): chỉ 1 GPU nên chạy tuần tự từng job.
  - Job state lưu in-memory (demo 1 process). Frontend poll GET /api/jobs/{id}.
  - Task là 1 callable nhận progress_cb(done, total, info) để cập nhật tiến độ.
=============================================================
"""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional


class JobManager:
    def __init__(self) -> None:
        self._jobs: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=1)

    # --------------------------------------------------------
    def submit(self, task: Callable[[Callable], dict], label: str = "") -> str:
        """task(progress_cb) -> result dict (JSON-serializable). Trả về job_id."""
        job_id = uuid.uuid4().hex[:12]
        now = time.time()
        with self._lock:
            self._jobs[job_id] = {
                "id": job_id,
                "status": "queued",
                "stage": "Đang chờ trong hàng đợi…",
                "label": label,
                "done": 0,
                "total": 0,
                "progress": 0.0,
                "last": None,
                "result": None,
                "error": None,
                "created_at": now,
                "updated_at": now,
            }
        self._executor.submit(self._run, job_id, task)
        return job_id

    # --------------------------------------------------------
    def _run(self, job_id: str, task: Callable[[Callable], dict]) -> None:
        self._update(job_id, status="running", stage="Đang khởi động…")

        def progress_cb(done: int, total: int, info: Optional[dict] = None) -> None:
            info = info or {}
            patch: dict = {"done": done, "total": total}
            if total > 0:
                patch["progress"] = round(min(done / total, 1.0), 4)
            if info.get("stage") == "computing_styles":
                patch["stage"] = "Đang trích style 2 giọng mẫu…"
            elif total > 0:
                patch["stage"] = f"Đang đọc câu {done}/{total}…"
                patch["last"] = {
                    "id": info.get("id"),
                    "role": info.get("role"),
                    "status": info.get("status"),
                    "text_preview": info.get("text_preview"),
                }
            self._update(job_id, **patch)

        try:
            result = task(progress_cb)
            self._update(
                job_id, status="done", stage="Hoàn tất ✓",
                progress=1.0, result=result,
            )
        except Exception as e:                           # noqa: BLE001
            self._update(
                job_id, status="error",
                stage="Lỗi", error=f"{type(e).__name__}: {e}",
            )

    # --------------------------------------------------------
    def _update(self, job_id: str, **patch) -> None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return
            job.update(patch)
            job["updated_at"] = time.time()

    def get(self, job_id: str) -> Optional[dict]:
        with self._lock:
            job = self._jobs.get(job_id)
            return dict(job) if job else None
