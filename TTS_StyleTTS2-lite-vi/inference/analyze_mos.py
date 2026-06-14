"""
=============================================================
  PHÂN TÍCH ĐIỂM MOS / SMOS — gộp phản hồi Google Form -> bảng kết quả
=============================================================
Đầu vào:
  --responses : CSV "long format" do bạn chuẩn hoá từ Google Form, có cột:
                  sample_file, mos[, smos][, rater_id]
                (xem inference/eval/MOS_FORM_GUIDE.md mục 4.1)
  --manifest  : output/eval/mos_manifest.csv (đáp án sample -> điều kiện thật)

Xuất:
  - In ra bảng MOS/SMOS theo từng điều kiện: trung bình ± CI 95%, N.
  - Ghi output/eval/mos_results.csv để dán vào báo cáo (Mục 5.2).

Chạy:
  python inference/analyze_mos.py --responses responses_long.csv
=============================================================
"""

from __future__ import annotations

import sys as _sys
for _s in (_sys.stdout, _sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import csv
import math
import re
import statistics
from pathlib import Path

INFERENCE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = INFERENCE_DIR.parent
DEFAULT_MANIFEST = PROJECT_ROOT / "output" / "eval" / "mos_manifest.csv"
DEFAULT_OUT = PROJECT_ROOT / "output" / "eval" / "mos_results.csv"

_SAMPLE_RE = re.compile(r"sample_\d+\.wav", re.IGNORECASE)


def _norm_sample(value: str) -> str | None:
    """Trích 'sample_037.wav' từ một ô bất kỳ (tên cột Form có thể dài dòng)."""
    if not value:
        return None
    m = _SAMPLE_RE.search(value)
    return m.group(0).lower() if m else None


def _to_float(v) -> float | None:
    try:
        x = float(str(v).strip())
        return x if not math.isnan(x) else None
    except (ValueError, TypeError):
        return None


def load_manifest(path: Path) -> dict[str, dict]:
    rows = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            key = _norm_sample(r["sample_file"])
            if key:
                rows[key] = r
    if not rows:
        raise ValueError(f"Manifest rỗng/không hợp lệ: {path}")
    return rows


def load_responses(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        # Tự dò tên cột
        sample_col = next((c for c in cols if "sample" in c.lower()), None)
        mos_col = next((c for c in cols if c.lower().strip() in
                        ("mos", "naturalness", "do_tu_nhien", "tu_nhien")), None)
        smos_col = next((c for c in cols if c.lower().strip() in
                         ("smos", "similarity", "tuong_dong", "do_tuong_dong")), None)
        if not sample_col or not mos_col:
            raise ValueError(
                "Responses CSV cần ít nhất 2 cột: 'sample_file' và 'mos'. "
                f"Đang thấy: {cols}"
            )
        out = []
        for r in reader:
            sf = _norm_sample(r.get(sample_col, ""))
            if not sf:
                continue
            out.append({
                "sample_file": sf,
                "mos": _to_float(r.get(mos_col)),
                "smos": _to_float(r.get(smos_col)) if smos_col else None,
            })
    return out


def ci95(vals: list[float]) -> tuple[float, float, float, int]:
    """Trả về (mean, std, half_width_CI95, N)."""
    n = len(vals)
    if n == 0:
        return (0.0, 0.0, 0.0, 0)
    mean = statistics.mean(vals)
    std = statistics.stdev(vals) if n > 1 else 0.0
    half = 1.96 * std / math.sqrt(n) if n > 1 else 0.0
    return (mean, std, half, n)


def main() -> None:
    ap = argparse.ArgumentParser(description="Phân tích MOS/SMOS từ Google Form")
    ap.add_argument("--responses", required=True, help="CSV long-format đã chuẩn hoá")
    ap.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    manifest = load_manifest(Path(args.manifest))
    responses = load_responses(Path(args.responses))
    if not responses:
        raise SystemExit("Không đọc được dòng phản hồi nào.")

    # Gom điểm theo điều kiện
    mos_by_cond: dict[str, list[float]] = {}
    smos_by_cond: dict[str, list[float]] = {}
    n_skipped = 0
    for r in responses:
        m = manifest.get(r["sample_file"])
        if m is None:
            n_skipped += 1
            continue
        cond = m["true_condition"]
        if r["mos"] is not None:
            mos_by_cond.setdefault(cond, []).append(r["mos"])
        if r["smos"] is not None and m.get("smos_eligible") == "yes":
            smos_by_cond.setdefault(cond, []).append(r["smos"])

    order = ["groundtruth", "final", "pretrained"]
    out_rows = []
    print("\n=== MOS — Độ tự nhiên (trung bình ± CI95) ===")
    for cond in order:
        if cond in mos_by_cond:
            mean, std, half, n = ci95(mos_by_cond[cond])
            print(f"  {cond:12s}: {mean:.2f} ± {half:.2f}  (std={std:.2f}, n={n})")
            out_rows.append({"metric": "MOS", "condition": cond,
                             "mean": round(mean, 3), "ci95": round(half, 3),
                             "std": round(std, 3), "n": n})

    print("\n=== SMOS — Độ tương đồng giọng (chỉ in-domain) ===")
    for cond in order:
        if cond in smos_by_cond:
            mean, std, half, n = ci95(smos_by_cond[cond])
            print(f"  {cond:12s}: {mean:.2f} ± {half:.2f}  (std={std:.2f}, n={n})")
            out_rows.append({"metric": "SMOS", "condition": cond,
                             "mean": round(mean, 3), "ci95": round(half, 3),
                             "std": round(std, 3), "n": n})

    if n_skipped:
        print(f"\n  (Bỏ qua {n_skipped} dòng không khớp manifest.)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "condition", "mean", "ci95", "std", "n"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"\n✅ Đã ghi: {args.out}")


if __name__ == "__main__":
    main()
