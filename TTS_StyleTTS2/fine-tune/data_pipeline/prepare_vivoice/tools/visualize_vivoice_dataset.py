# -*- coding: utf-8 -*-
"""
visualize_vivoice_dataset.py

Tạo bộ biểu đồ + report thống kê data gốc và data sau khi lọc cho pipeline ViVoice / StyleTTS2.

Script này KHÔNG sửa/xóa data. Chỉ đọc:
  - output/speaker_id_map.json.backup_before_filter  : speaker map gốc trước step6b
  - output/speaker_id_map.json                       : speaker map sau lọc step6b
  - output/vivoice_train_list.txt, output/vivoice_val_list.txt : filelist sau lọc, optional
  - output/phoneme_vocab.json hoặc phoneme_vocab.json           : optional

Các biểu đồ được lưu vào:
  output/vivoice_visualizations/

Cài thư viện nếu thiếu:
  pip install matplotlib numpy

Chạy nhanh từ folder prepare_vivoice:
  python visualize_vivoice_dataset.py

Chạy tự chỉ định path:
  python visualize_vivoice_dataset.py ^
    --original-map "D:/.../output/speaker_id_map.json.backup_before_filter" ^
    --filtered-map "D:/.../output/speaker_id_map.json" ^
    --train-list "D:/.../output/vivoice_train_list.txt" ^
    --val-list "D:/.../output/vivoice_val_list.txt" ^
    --phoneme-vocab "D:/.../output/phoneme_vocab.json" ^
    --out-dir "D:/.../output/vivoice_visualizations"
"""

import os
import sys
import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Optional, Any

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
except Exception as e:
    raise RuntimeError(
        "Không import được matplotlib. Hãy cài bằng: pip install matplotlib numpy"
    ) from e


# =============================================================================
# Windows UTF-8 fix
# =============================================================================

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"


# =============================================================================
# Default paths
# =============================================================================

DEFAULT_BASE_DIR = Path(
    r"D:/Documents/HUST/HUST_Project/Project_Final/TTS_StyleTTS2"
    r"/fine-tune/data_pipeline/prepare_vivoice"
)

DEFAULT_OUTPUT_DIR = DEFAULT_BASE_DIR / "output"

DEFAULT_ORIGINAL_MAP = DEFAULT_OUTPUT_DIR / "speaker_id_map.json.backup_before_filter"
DEFAULT_FILTERED_MAP = DEFAULT_OUTPUT_DIR / "speaker_id_map.json"
DEFAULT_TRAIN_LIST = DEFAULT_OUTPUT_DIR / "vivoice_train_list.txt"
DEFAULT_VAL_LIST = DEFAULT_OUTPUT_DIR / "vivoice_val_list.txt"

# Có project đặt phoneme_vocab.json ở output, có project đặt ngay prepare_vivoice.
DEFAULT_PHONEME_VOCAB_1 = DEFAULT_OUTPUT_DIR / "phoneme_vocab.json"
DEFAULT_PHONEME_VOCAB_2 = DEFAULT_BASE_DIR / "phoneme_vocab.json"

DEFAULT_OUT_DIR = DEFAULT_OUTPUT_DIR / "vivoice_visualizations"


# =============================================================================
# Modern palette — lấy cảm hứng từ ảnh palette bạn gửi
# =============================================================================

COLORS = {
    "navy": "#001C44",
    "deep": "#0C5776",
    "teal": "#2D99AE",
    "mint": "#BCFFFE",
    "peach": "#F8DAD0",
    "bg": "#F7FBFC",
    "panel": "#FFFFFF",
    "text": "#17202A",
    "muted": "#6B7280",
    "grid": "#D7DEE5",
    "danger": "#E76F51",
    "warning": "#F4A261",
    "ok": "#2A9D8F",
    "purple": "#7469B6",
    "rose": "#D76C82",
}

PALETTE = [
    COLORS["navy"],
    COLORS["deep"],
    COLORS["teal"],
    COLORS["mint"],
    COLORS["peach"],
    COLORS["purple"],
    COLORS["rose"],
    COLORS["warning"],
]


# =============================================================================
# Generic helpers
# =============================================================================

def fmt_int(x: float) -> str:
    try:
        return f"{int(round(x)):,}"
    except Exception:
        return str(x)


def fmt_pct(x: float, digits: int = 1) -> str:
    if x is None or math.isnan(x):
        return "N/A"
    return f"{x:.{digits}f}%"


def human_num(x: float, pos: Optional[int] = None) -> str:
    try:
        x = float(x)
    except Exception:
        return str(x)

    abs_x = abs(x)
    if abs_x >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs_x >= 1_000:
        return f"{x / 1_000:.0f}K"
    return f"{x:.0f}"


def safe_div(a: float, b: float) -> float:
    return 0.0 if b == 0 else (a / b)


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy JSON: {path}")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def maybe_load_json(path: Optional[Path]) -> Optional[Dict[str, Any]]:
    if path is None:
        return None
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def ensure_out_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def setup_plot_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "savefig.facecolor": COLORS["bg"],
        "axes.edgecolor": "#E5E7EB",
        "axes.labelcolor": COLORS["text"],
        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],
        "text.color": COLORS["text"],
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 17,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "legend.frameon": False,
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.35,
    })


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def clean_axis(ax, grid_axis: str = "y") -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#E5E7EB")
    ax.spines["bottom"].set_color("#E5E7EB")
    ax.grid(True, axis=grid_axis)
    ax.set_axisbelow(True)
    ax.yaxis.set_major_formatter(FuncFormatter(human_num))


def add_bar_labels(ax, bars, orient: str = "v", fontsize: int = 9, color: Optional[str] = None) -> None:
    color = color or COLORS["muted"]
    if orient == "v":
        for b in bars:
            h = b.get_height()
            ax.text(
                b.get_x() + b.get_width() / 2,
                h,
                human_num(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
                color=color,
            )
    else:
        for b in bars:
            w = b.get_width()
            ax.text(
                w,
                b.get_y() + b.get_height() / 2,
                human_num(w),
                ha="left",
                va="center",
                fontsize=fontsize,
                color=color,
            )


def annotate_note(fig, text: str) -> None:
    fig.text(0.01, 0.01, text, fontsize=9, color=COLORS["muted"])


# =============================================================================
# Data loading / parsing
# =============================================================================

def counts_from_map(data: Dict[str, Any]) -> Dict[int, int]:
    raw = data.get("speaker_id_record_counts", {})
    out: Dict[int, int] = {}
    for k, v in raw.items():
        try:
            out[int(k)] = int(v)
        except Exception:
            continue
    return out


def sid_to_channel(data: Dict[str, Any]) -> Dict[int, str]:
    """
    speaker_id_to_channels thường có format:
      "1": ["@VoizFM"]
    """
    raw = data.get("speaker_id_to_channels", {})
    out: Dict[int, str] = {}
    for k, v in raw.items():
        try:
            sid = int(k)
        except Exception:
            continue
        if isinstance(v, list) and v:
            out[sid] = str(v[0])
        elif isinstance(v, str):
            out[sid] = v
        else:
            out[sid] = f"speaker_{sid}"
    return out


def meta_total_records(data: Dict[str, Any], counts: Dict[int, int]) -> int:
    meta = data.get("_metadata", {})
    train = meta.get("train_records")
    val = meta.get("val_records")
    if isinstance(train, int) and isinstance(val, int):
        return train + val
    return sum(counts.values())


def metadata_summary(label: str, data: Dict[str, Any], counts: Dict[int, int]) -> Dict[str, Any]:
    meta = data.get("_metadata", {})
    values = list(counts.values())
    total = meta_total_records(data, counts)
    return {
        "label": label,
        "records": total,
        "speakers": len(counts),
        "train_records": meta.get("train_records"),
        "val_records": meta.get("val_records"),
        "min_samples_per_speaker": meta.get("min_samples_per_speaker"),
        "cap_per_speaker": meta.get("cap_per_speaker"),
        "mean_records_per_speaker": statistics.mean(values) if values else 0,
        "median_records_per_speaker": statistics.median(values) if values else 0,
        "max_records_per_speaker": max(values) if values else 0,
        "min_records_per_speaker": min(values) if values else 0,
    }


def build_speaker_comparison(
    original_map: Dict[str, Any],
    filtered_map: Dict[str, Any],
) -> List[Dict[str, Any]]:
    original_counts = counts_from_map(original_map)
    filtered_counts = counts_from_map(filtered_map)

    channel_lookup = sid_to_channel(original_map)
    channel_lookup.update(sid_to_channel(filtered_map))

    all_sids = sorted(set(original_counts) | set(filtered_counts))
    rows: List[Dict[str, Any]] = []

    for sid in all_sids:
        orig = int(original_counts.get(sid, 0))
        filt = int(filtered_counts.get(sid, 0))
        removed = max(orig - filt, 0)
        retained_pct = safe_div(filt, orig) * 100 if orig else 0.0

        if orig > 0 and filt > 0:
            status = "kept"
        elif orig > 0 and filt == 0:
            status = "dropped"
        elif orig == 0 and filt > 0:
            status = "new_in_filtered"
        else:
            status = "none"

        rows.append({
            "speaker_id": sid,
            "channel": channel_lookup.get(sid, f"speaker_{sid}"),
            "original_records": orig,
            "filtered_records": filt,
            "removed_records": removed,
            "retained_pct": retained_pct,
            "status": status,
        })
    return rows


def parse_filelist_line(line: str, delimiter: str = "|") -> Optional[Tuple[str, str, int]]:
    """
    Format:
      wav_path|phoneme|speaker_id

    Dùng rsplit để speaker_id là phần cuối.
    Phần bên trái split lần đầu để lấy wav_path + phoneme.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None

    parts = line.rsplit(delimiter, 1)
    if len(parts) != 2:
        return None

    try:
        sid = int(parts[1])
    except ValueError:
        return None

    left = parts[0].split(delimiter, 1)
    if len(left) != 2:
        return None

    wav_path, phoneme = left[0].strip(), left[1].strip()
    return wav_path, phoneme, sid


def load_filelist_stats(path: Path, split_name: str, delimiter: str = "|") -> Dict[str, Any]:
    stats = {
        "split": split_name,
        "path": str(path),
        "exists": path.exists(),
        "records": 0,
        "bad_lines": 0,
        "speaker_counts": Counter(),
        "phoneme_char_lengths": [],
        "phoneme_token_lengths": [],
        "wav_paths": [],
    }
    if not path.exists():
        return stats

    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            parsed = parse_filelist_line(raw, delimiter=delimiter)
            if parsed is None:
                if raw.strip() and not raw.strip().startswith("#"):
                    stats["bad_lines"] += 1
                continue

            wav_path, phoneme, sid = parsed
            stats["records"] += 1
            stats["speaker_counts"][sid] += 1
            stats["phoneme_char_lengths"].append(len(phoneme))
            stats["phoneme_token_lengths"].append(len(phoneme.split()))
            stats["wav_paths"].append(wav_path)

    return stats


def combine_filelist_stats(train_stats: Dict[str, Any], val_stats: Dict[str, Any]) -> Dict[str, Any]:
    combined_counts = Counter()
    combined_counts.update(train_stats.get("speaker_counts", Counter()))
    combined_counts.update(val_stats.get("speaker_counts", Counter()))

    return {
        "records": train_stats.get("records", 0) + val_stats.get("records", 0),
        "bad_lines": train_stats.get("bad_lines", 0) + val_stats.get("bad_lines", 0),
        "speaker_counts": combined_counts,
        "phoneme_char_lengths": train_stats.get("phoneme_char_lengths", []) + val_stats.get("phoneme_char_lengths", []),
        "phoneme_token_lengths": train_stats.get("phoneme_token_lengths", []) + val_stats.get("phoneme_token_lengths", []),
    }


# =============================================================================
# CSV / JSON exports
# =============================================================================

def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =============================================================================
# Chart functions
# =============================================================================

def chart_dashboard(
    out_dir: Path,
    original_summary: Dict[str, Any],
    filtered_summary: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    orig_records = original_summary["records"]
    filt_records = filtered_summary["records"]
    orig_speakers = original_summary["speakers"]
    filt_speakers = filtered_summary["speakers"]

    removed_records = orig_records - filt_records
    dropped_speakers = sum(1 for r in rows if r["status"] == "dropped")
    kept_speakers = sum(1 for r in rows if r["status"] == "kept")
    retained_pct = safe_div(filt_records, orig_records) * 100

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("ViVoice Dataset Overview — Before vs After Filtering", x=0.03, ha="left",
                 fontsize=24, fontweight="bold", color=COLORS["navy"])

    card_specs = [
        ("Original records", fmt_int(orig_records), COLORS["deep"]),
        ("Filtered records", fmt_int(filt_records), COLORS["teal"]),
        ("Removed records", fmt_int(removed_records), COLORS["danger"]),
        ("Retention", fmt_pct(retained_pct), COLORS["ok"]),
        ("Original speakers", fmt_int(orig_speakers), COLORS["deep"]),
        ("Filtered speakers", fmt_int(filt_speakers), COLORS["teal"]),
        ("Dropped speakers", fmt_int(dropped_speakers), COLORS["danger"]),
        ("Kept speakers", fmt_int(kept_speakers), COLORS["ok"]),
    ]

    for i, (title, value, color) in enumerate(card_specs):
        row = i // 4
        col = i % 4
        ax = fig.add_axes([0.04 + col * 0.24, 0.68 - row * 0.20, 0.21, 0.14])
        ax.set_facecolor(COLORS["panel"])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(0.05, 0.72, title, fontsize=11, color=COLORS["muted"], transform=ax.transAxes)
        ax.text(0.05, 0.25, value, fontsize=24, fontweight="bold", color=color, transform=ax.transAxes)
        ax.axvline(0.0, ymin=0.15, ymax=0.85, color=color, linewidth=5, alpha=0.9)

    # Main bars
    ax1 = fig.add_axes([0.07, 0.10, 0.38, 0.36])
    vals = [orig_records, filt_records, removed_records]
    labels = ["Original", "Filtered", "Removed"]
    bars = ax1.bar(labels, vals, color=[COLORS["deep"], COLORS["teal"], COLORS["danger"]], width=0.58)
    clean_axis(ax1)
    add_bar_labels(ax1, bars)
    ax1.set_title("Record Volume")
    ax1.set_ylabel("Records")

    # Speaker status donut
    ax2 = fig.add_axes([0.56, 0.08, 0.36, 0.40])
    sizes = [kept_speakers, dropped_speakers]
    labels = ["Kept", "Dropped"]
    wedges, _ = ax2.pie(
        sizes,
        startangle=90,
        colors=[COLORS["teal"], COLORS["peach"]],
        wedgeprops=dict(width=0.42, edgecolor=COLORS["bg"], linewidth=3),
    )
    ax2.text(0, 0.04, f"{filt_speakers}/{orig_speakers}", ha="center", va="center",
             fontsize=24, fontweight="bold", color=COLORS["navy"])
    ax2.text(0, -0.16, "speakers kept", ha="center", va="center",
             fontsize=11, color=COLORS["muted"])
    ax2.legend(wedges, labels, loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax2.set_title("Speaker Retention")

    annotate_note(fig, "Generated by visualize_vivoice_dataset.py")
    save_fig(fig, out_dir / "00_dashboard_overview.png")


def chart_total_records_speakers(out_dir: Path, original_summary: Dict[str, Any], filtered_summary: Dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    labels = ["Original", "Filtered"]
    records = [original_summary["records"], filtered_summary["records"]]
    speakers = [original_summary["speakers"], filtered_summary["speakers"]]

    ax = axes[0]
    bars = ax.bar(labels, records, color=[COLORS["deep"], COLORS["teal"]], width=0.55)
    clean_axis(ax)
    add_bar_labels(ax, bars)
    ax.set_title("Total Records")
    ax.set_ylabel("Records")

    ax = axes[1]
    bars = ax.bar(labels, speakers, color=[COLORS["deep"], COLORS["teal"]], width=0.55)
    clean_axis(ax)
    add_bar_labels(ax, bars)
    ax.set_title("Unique Speakers")
    ax.set_ylabel("Speakers")

    fig.suptitle("Dataset Size Comparison", fontsize=20, fontweight="bold", color=COLORS["navy"])
    save_fig(fig, out_dir / "01_total_records_and_speakers.png")


def chart_train_val_split(out_dir: Path, original_summary: Dict[str, Any], filtered_summary: Dict[str, Any]) -> None:
    datasets = [original_summary, filtered_summary]
    labels = ["Original", "Filtered"]

    train_vals = []
    val_vals = []
    for s in datasets:
        train_vals.append(s.get("train_records") or 0)
        val_vals.append(s.get("val_records") or 0)

    if sum(train_vals) + sum(val_vals) == 0:
        return

    x = np.arange(len(labels))
    width = 0.56

    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(x, train_vals, width, label="Train", color=COLORS["deep"])
    b2 = ax.bar(x, val_vals, width, bottom=train_vals, label="Val", color=COLORS["mint"], edgecolor=COLORS["deep"], linewidth=0.6)

    clean_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Train / Validation Split")
    ax.set_ylabel("Records")
    ax.legend()

    for idx, (tr, va) in enumerate(zip(train_vals, val_vals)):
        total = tr + va
        ax.text(idx, total, human_num(total), ha="center", va="bottom", fontsize=10, color=COLORS["muted"])

    save_fig(fig, out_dir / "02_train_val_split.png")


def chart_speaker_distribution_sorted(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    sorted_rows = sorted(rows, key=lambda r: r["original_records"], reverse=True)
    x = np.arange(len(sorted_rows))
    orig = np.array([r["original_records"] for r in sorted_rows])
    filt = np.array([r["filtered_records"] for r in sorted_rows])

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(x, orig, color=COLORS["deep"], linewidth=2.2, label="Original")
    ax.fill_between(x, orig, color=COLORS["deep"], alpha=0.08)
    ax.plot(x, filt, color=COLORS["teal"], linewidth=2.2, label="Filtered")
    ax.fill_between(x, filt, color=COLORS["teal"], alpha=0.12)

    clean_axis(ax)
    ax.set_title("Speaker Record Distribution — Sorted by Original Count")
    ax.set_xlabel("Speakers sorted by original record count")
    ax.set_ylabel("Records / speaker")
    ax.legend()

    save_fig(fig, out_dir / "03_speaker_distribution_sorted.png")


def chart_top_speakers(out_dir: Path, rows: List[Dict[str, Any]], top_n: int) -> None:
    top = sorted(rows, key=lambda r: r["original_records"], reverse=True)[:top_n]
    if not top:
        return

    labels = [f'{r["speaker_id"]}\n{r["channel"][:18]}' for r in top]
    orig = np.array([r["original_records"] for r in top])
    filt = np.array([r["filtered_records"] for r in top])

    y = np.arange(len(top))
    h = 0.38

    fig, ax = plt.subplots(figsize=(16, max(7, top_n * 0.43)))
    ax.barh(y + h / 2, orig, height=h, color=COLORS["deep"], label="Original")
    ax.barh(y - h / 2, filt, height=h, color=COLORS["teal"], label="Filtered")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    clean_axis(ax, grid_axis="x")
    ax.set_title(f"Top {top_n} Speakers — Original vs Filtered")
    ax.set_xlabel("Records")
    ax.legend()

    save_fig(fig, out_dir / f"04_top_{top_n}_speakers_before_after.png")


def chart_removed_records(out_dir: Path, rows: List[Dict[str, Any]], top_n: int) -> None:
    top = sorted(rows, key=lambda r: r["removed_records"], reverse=True)[:top_n]
    top = [r for r in top if r["removed_records"] > 0]
    if not top:
        return

    labels = [f'{r["speaker_id"]}\n{r["channel"][:18]}' for r in top]
    removed = np.array([r["removed_records"] for r in top])
    retained_pct = [r["retained_pct"] for r in top]
    y = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(16, max(7, top_n * 0.43)))
    bars = ax.barh(y, removed, color=COLORS["danger"], alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    clean_axis(ax, grid_axis="x")
    ax.set_title(f"Top {top_n} Speakers by Removed Records")
    ax.set_xlabel("Removed records")

    for b, p in zip(bars, retained_pct):
        ax.text(
            b.get_width(),
            b.get_y() + b.get_height() / 2,
            f"  kept {p:.1f}%",
            va="center",
            fontsize=9,
            color=COLORS["muted"],
        )

    save_fig(fig, out_dir / f"05_top_{top_n}_removed_records.png")


def chart_retention_hist(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    vals = [r["retained_pct"] for r in rows if r["original_records"] > 0]

    fig, ax = plt.subplots(figsize=(12, 6))
    bins = np.linspace(0, 100, 21)
    ax.hist(vals, bins=bins, color=COLORS["teal"], alpha=0.90, edgecolor="white")
    clean_axis(ax)
    ax.set_title("Retention Rate Distribution by Speaker")
    ax.set_xlabel("Retained % after filtering")
    ax.set_ylabel("Number of speakers")
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))

    save_fig(fig, out_dir / "06_retention_rate_histogram.png")


def chart_scatter_original_filtered(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    orig = np.array([r["original_records"] for r in rows])
    filt = np.array([r["filtered_records"] for r in rows])
    status = [r["status"] for r in rows]

    colors = [COLORS["teal"] if s == "kept" else COLORS["peach"] for s in status]

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.scatter(orig, filt, s=52, c=colors, alpha=0.88, edgecolor="white", linewidth=0.6)

    max_val = max(orig.max() if len(orig) else 0, filt.max() if len(filt) else 0)
    ax.plot([0, max_val], [0, max_val], linestyle="--", color=COLORS["muted"], linewidth=1.2, label="No filtering line")

    clean_axis(ax)
    ax.set_title("Original vs Filtered Records per Speaker")
    ax.set_xlabel("Original records")
    ax.set_ylabel("Filtered records")
    ax.legend()

    save_fig(fig, out_dir / "07_original_vs_filtered_scatter.png")


def chart_distribution_hist(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    orig = np.array([r["original_records"] for r in rows if r["original_records"] > 0])
    filt = np.array([r["filtered_records"] for r in rows if r["filtered_records"] > 0])

    if len(orig) == 0:
        return

    max_val = max(orig.max(), filt.max() if len(filt) else 0)
    # Dữ liệu lệch lớn nên dùng bins logspace.
    bins = np.unique(np.logspace(0, math.log10(max_val + 1), 28).astype(int))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.hist(orig, bins=bins, color=COLORS["deep"], alpha=0.65, label="Original")
    if len(filt):
        ax.hist(filt, bins=bins, color=COLORS["teal"], alpha=0.70, label="Filtered")

    ax.set_xscale("log")
    clean_axis(ax)
    ax.set_title("Records per Speaker Distribution")
    ax.set_xlabel("Records per speaker — log scale")
    ax.set_ylabel("Number of speakers")
    ax.legend()

    save_fig(fig, out_dir / "08_records_per_speaker_histogram_log.png")


def chart_pareto(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    def cum_pct(values: List[int]) -> np.ndarray:
        arr = np.array(sorted(values, reverse=True), dtype=float)
        if arr.sum() == 0:
            return arr
        return np.cumsum(arr) / arr.sum() * 100

    orig_vals = [r["original_records"] for r in rows if r["original_records"] > 0]
    filt_vals = [r["filtered_records"] for r in rows if r["filtered_records"] > 0]

    fig, ax = plt.subplots(figsize=(13, 7))
    ax.plot(np.arange(1, len(orig_vals) + 1), cum_pct(orig_vals), color=COLORS["deep"], linewidth=2.4, label="Original")
    ax.plot(np.arange(1, len(filt_vals) + 1), cum_pct(filt_vals), color=COLORS["teal"], linewidth=2.4, label="Filtered")

    ax.axhline(80, color=COLORS["danger"], linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(1, 81.5, "80% records", color=COLORS["danger"], fontsize=10)

    clean_axis(ax)
    ax.set_title("Cumulative Record Share — Pareto View")
    ax.set_xlabel("Top N speakers")
    ax.set_ylabel("Cumulative % of records")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0f}%"))
    ax.legend()

    save_fig(fig, out_dir / "09_pareto_cumulative_records.png")


def chart_kept_dropped(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    kept = sum(1 for r in rows if r["status"] == "kept")
    dropped = sum(1 for r in rows if r["status"] == "dropped")
    new = sum(1 for r in rows if r["status"] == "new_in_filtered")

    labels = []
    sizes = []
    colors = []
    if kept:
        labels.append("Kept")
        sizes.append(kept)
        colors.append(COLORS["teal"])
    if dropped:
        labels.append("Dropped")
        sizes.append(dropped)
        colors.append(COLORS["peach"])
    if new:
        labels.append("New in filtered")
        sizes.append(new)
        colors.append(COLORS["purple"])

    if not sizes:
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, _ = ax.pie(
        sizes,
        startangle=90,
        colors=colors,
        wedgeprops=dict(width=0.42, edgecolor=COLORS["bg"], linewidth=3),
    )
    total = sum(sizes)
    ax.text(0, 0.03, fmt_int(total), ha="center", va="center", fontsize=28, fontweight="bold", color=COLORS["navy"])
    ax.text(0, -0.16, "speakers", ha="center", va="center", fontsize=12, color=COLORS["muted"])
    ax.legend(wedges, [f"{l}: {s}" for l, s in zip(labels, sizes)], loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    ax.set_title("Kept vs Dropped Speakers")

    save_fig(fig, out_dir / "10_kept_vs_dropped_speakers_donut.png")


def chart_boxplot(out_dir: Path, rows: List[Dict[str, Any]]) -> None:
    orig = [r["original_records"] for r in rows if r["original_records"] > 0]
    filt = [r["filtered_records"] for r in rows if r["filtered_records"] > 0]
    if not orig or not filt:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bp = ax.boxplot(
        [orig, filt],
        labels=["Original", "Filtered"],
        patch_artist=True,
        showfliers=True,
        medianprops=dict(color=COLORS["navy"], linewidth=2),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
        flierprops=dict(marker="o", markersize=5, markerfacecolor=COLORS["peach"], markeredgecolor="white", alpha=0.8),
    )
    for patch, color in zip(bp["boxes"], [COLORS["deep"], COLORS["teal"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_yscale("log")
    clean_axis(ax)
    ax.set_title("Speaker Count Spread — Boxplot")
    ax.set_ylabel("Records per speaker — log scale")

    save_fig(fig, out_dir / "11_speaker_count_boxplot_log.png")


def chart_phoneme_vocab(out_dir: Path, phoneme_vocab: Optional[Dict[str, Any]], top_n: int) -> None:
    if not phoneme_vocab:
        return

    freqs = phoneme_vocab.get("char_frequencies", {})
    if not isinstance(freqs, dict) or not freqs:
        return

    items = []
    for ch, c in freqs.items():
        try:
            items.append((ch, int(c)))
        except Exception:
            continue
    if not items:
        return

    top = sorted(items, key=lambda x: x[1], reverse=True)[:top_n]
    labels = [repr(ch)[1:-1] if ch.strip() == "" else ch for ch, _ in top]
    vals = [v for _, v in top]

    fig, ax = plt.subplots(figsize=(15, max(7, top_n * 0.28)))
    y = np.arange(len(top))
    bars = ax.barh(y, vals, color=COLORS["deep"])
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    clean_axis(ax, grid_axis="x")
    ax.set_title(f"Top {top_n} Phoneme / Character Frequencies")
    ax.set_xlabel("Frequency")
    add_bar_labels(ax, bars, orient="h", fontsize=8)

    save_fig(fig, out_dir / f"12_top_{top_n}_phoneme_char_frequencies.png")

    rare = sorted(items, key=lambda x: x[1])[:top_n]
    labels = [repr(ch)[1:-1] if ch.strip() == "" else ch for ch, _ in rare]
    vals = [v for _, v in rare]

    fig, ax = plt.subplots(figsize=(15, max(7, top_n * 0.28)))
    y = np.arange(len(rare))
    bars = ax.barh(y, vals, color=COLORS["peach"], edgecolor=COLORS["danger"], linewidth=0.6)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    clean_axis(ax, grid_axis="x")
    ax.set_title(f"Rare {top_n} Phoneme / Character Frequencies")
    ax.set_xlabel("Frequency")
    add_bar_labels(ax, bars, orient="h", fontsize=8)

    save_fig(fig, out_dir / f"13_rare_{top_n}_phoneme_char_frequencies.png")


def chart_filelist_lengths(
    out_dir: Path,
    train_stats: Dict[str, Any],
    val_stats: Dict[str, Any],
    combined_stats: Dict[str, Any],
) -> None:
    if not train_stats.get("exists") and not val_stats.get("exists"):
        return

    token_lengths = combined_stats.get("phoneme_token_lengths", [])
    char_lengths = combined_stats.get("phoneme_char_lengths", [])

    if token_lengths:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.hist(token_lengths, bins=60, color=COLORS["teal"], alpha=0.85, edgecolor="white")
        clean_axis(ax)
        ax.set_title("Phoneme Token Length Distribution — Current Filelists")
        ax.set_xlabel("Number of phoneme tokens per record")
        ax.set_ylabel("Records")
        save_fig(fig, out_dir / "14_filelist_phoneme_token_length_histogram.png")

    if char_lengths:
        fig, ax = plt.subplots(figsize=(13, 6))
        ax.hist(char_lengths, bins=60, color=COLORS["deep"], alpha=0.85, edgecolor="white")
        clean_axis(ax)
        ax.set_title("Phoneme Character Length Distribution — Current Filelists")
        ax.set_xlabel("Number of characters in phoneme string")
        ax.set_ylabel("Records")
        save_fig(fig, out_dir / "15_filelist_phoneme_char_length_histogram.png")

    # Train/val speaker distribution
    train_counts: Counter = train_stats.get("speaker_counts", Counter())
    val_counts: Counter = val_stats.get("speaker_counts", Counter())
    all_sids = sorted(set(train_counts) | set(val_counts))
    if not all_sids:
        return

    train_vals = np.array([train_counts.get(sid, 0) for sid in all_sids])
    val_vals = np.array([val_counts.get(sid, 0) for sid in all_sids])
    x = np.arange(len(all_sids))

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.bar(x, train_vals, color=COLORS["deep"], label="Train", width=0.82)
    ax.bar(x, val_vals, bottom=train_vals, color=COLORS["mint"], edgecolor=COLORS["deep"], linewidth=0.2, label="Val", width=0.82)
    clean_axis(ax)
    ax.set_title("Current Train/Val Records by Speaker ID")
    ax.set_xlabel("Speaker ID")
    ax.set_ylabel("Records")
    ax.legend()
    # Không label hết nếu nhiều speaker
    step = max(1, len(all_sids) // 20)
    ax.set_xticks(x[::step])
    ax.set_xticklabels([str(sid) for sid in all_sids[::step]], rotation=0)

    save_fig(fig, out_dir / "16_filelist_train_val_by_speaker.png")


def chart_cap_effect(out_dir: Path, rows: List[Dict[str, Any]], filtered_map: Dict[str, Any]) -> None:
    meta = filtered_map.get("_metadata", {})
    min_samples = meta.get("min_samples_per_speaker")
    cap = meta.get("cap_per_speaker")

    if not isinstance(min_samples, int) and not isinstance(cap, int):
        return

    sorted_rows = sorted(rows, key=lambda r: r["original_records"], reverse=True)
    x = np.arange(len(sorted_rows))
    orig = np.array([r["original_records"] for r in sorted_rows])
    filt = np.array([r["filtered_records"] for r in sorted_rows])

    fig, ax = plt.subplots(figsize=(18, 7))
    ax.plot(x, orig, color=COLORS["deep"], linewidth=2.2, label="Original")
    ax.plot(x, filt, color=COLORS["teal"], linewidth=2.2, label="Filtered")

    if isinstance(min_samples, int):
        ax.axhline(min_samples, linestyle="--", color=COLORS["warning"], linewidth=1.4, label=f"min_samples={min_samples:,}")
    if isinstance(cap, int) and cap > 0:
        ax.axhline(cap, linestyle="--", color=COLORS["danger"], linewidth=1.4, label=f"cap={cap:,}")

    clean_axis(ax)
    ax.set_title("Filter Threshold and Cap Effect")
    ax.set_xlabel("Speakers sorted by original records")
    ax.set_ylabel("Records / speaker")
    ax.legend()

    save_fig(fig, out_dir / "17_filter_threshold_and_cap_effect.png")


def chart_record_loss_waterfall(out_dir: Path, original_summary: Dict[str, Any], filtered_summary: Dict[str, Any]) -> None:
    original = original_summary["records"]
    filtered = filtered_summary["records"]
    removed = original - filtered

    labels = ["Original", "Removed", "Filtered"]
    values = [original, -removed, filtered]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(labels[0], original, color=COLORS["deep"])
    ax.bar(labels[1], removed, bottom=filtered, color=COLORS["danger"])
    ax.bar(labels[2], filtered, color=COLORS["teal"])

    ax.plot([0, 1], [original, original], color=COLORS["muted"], linestyle=":", linewidth=1)
    ax.plot([1, 2], [filtered, filtered], color=COLORS["muted"], linestyle=":", linewidth=1)

    clean_axis(ax)
    ax.set_title("Record Reduction Waterfall")
    ax.set_ylabel("Records")

    ax.text(0, original, fmt_int(original), ha="center", va="bottom", fontsize=10)
    ax.text(1, filtered + removed / 2, f"-{fmt_int(removed)}", ha="center", va="center", fontsize=12, color="white", fontweight="bold")
    ax.text(2, filtered, fmt_int(filtered), ha="center", va="bottom", fontsize=10)

    save_fig(fig, out_dir / "18_record_reduction_waterfall.png")


# =============================================================================
# Main
# =============================================================================

def resolve_phoneme_vocab_path(raw: Optional[str]) -> Optional[Path]:
    if raw:
        p = Path(raw)
        return p if p.exists() else p

    if DEFAULT_PHONEME_VOCAB_1.exists():
        return DEFAULT_PHONEME_VOCAB_1
    if DEFAULT_PHONEME_VOCAB_2.exists():
        return DEFAULT_PHONEME_VOCAB_2
    return None


def run(args: argparse.Namespace) -> int:
    setup_plot_style()

    original_map_path = Path(args.original_map)
    filtered_map_path = Path(args.filtered_map)
    train_list_path = Path(args.train_list) if args.train_list else None
    val_list_path = Path(args.val_list) if args.val_list else None
    phoneme_vocab_path = resolve_phoneme_vocab_path(args.phoneme_vocab)
    out_dir = Path(args.out_dir)

    ensure_out_dir(out_dir)

    print("=" * 88)
    print("ViVoice Dataset Visualization")
    print("=" * 88)
    print(f"Original speaker map : {original_map_path}")
    print(f"Filtered speaker map : {filtered_map_path}")
    print(f"Train list           : {train_list_path}")
    print(f"Val list             : {val_list_path}")
    print(f"Phoneme vocab        : {phoneme_vocab_path}")
    print(f"Output dir           : {out_dir}")
    print("")

    original_map = load_json(original_map_path)
    filtered_map = load_json(filtered_map_path)
    phoneme_vocab = maybe_load_json(phoneme_vocab_path) if phoneme_vocab_path else None

    original_counts = counts_from_map(original_map)
    filtered_counts = counts_from_map(filtered_map)

    if not original_counts:
        raise ValueError(f"Không thấy speaker_id_record_counts trong original map: {original_map_path}")
    if not filtered_counts:
        raise ValueError(f"Không thấy speaker_id_record_counts trong filtered map: {filtered_map_path}")

    rows = build_speaker_comparison(original_map, filtered_map)
    original_summary = metadata_summary("original", original_map, original_counts)
    filtered_summary = metadata_summary("filtered", filtered_map, filtered_counts)

    # Optional filelist stats
    empty_stats = {
        "exists": False,
        "records": 0,
        "bad_lines": 0,
        "speaker_counts": Counter(),
        "phoneme_char_lengths": [],
        "phoneme_token_lengths": [],
        "wav_paths": [],
    }
    train_stats = load_filelist_stats(train_list_path, "train", args.delimiter) if train_list_path else empty_stats
    val_stats = load_filelist_stats(val_list_path, "val", args.delimiter) if val_list_path else empty_stats
    combined_stats = combine_filelist_stats(train_stats, val_stats)

    # Derived summary
    orig_total = original_summary["records"]
    filt_total = filtered_summary["records"]
    removed_records = orig_total - filt_total
    kept_speakers = sum(1 for r in rows if r["status"] == "kept")
    dropped_speakers = sum(1 for r in rows if r["status"] == "dropped")

    summary = {
        "original": original_summary,
        "filtered": filtered_summary,
        "comparison": {
            "removed_records": removed_records,
            "retained_records_pct": safe_div(filt_total, orig_total) * 100,
            "dropped_speakers": dropped_speakers,
            "kept_speakers": kept_speakers,
            "retained_speakers_pct": safe_div(kept_speakers, original_summary["speakers"]) * 100,
        },
        "filelists_current": {
            "train_exists": train_stats.get("exists", False),
            "val_exists": val_stats.get("exists", False),
            "train_records": train_stats.get("records", 0),
            "val_records": val_stats.get("records", 0),
            "total_records": combined_stats.get("records", 0),
            "bad_lines": combined_stats.get("bad_lines", 0),
            "unique_speakers": len(combined_stats.get("speaker_counts", {})),
            "phoneme_token_length_mean": statistics.mean(combined_stats["phoneme_token_lengths"]) if combined_stats.get("phoneme_token_lengths") else None,
            "phoneme_token_length_median": statistics.median(combined_stats["phoneme_token_lengths"]) if combined_stats.get("phoneme_token_lengths") else None,
            "phoneme_char_length_mean": statistics.mean(combined_stats["phoneme_char_lengths"]) if combined_stats.get("phoneme_char_lengths") else None,
            "phoneme_char_length_median": statistics.median(combined_stats["phoneme_char_lengths"]) if combined_stats.get("phoneme_char_lengths") else None,
        },
    }

    # Write data reports
    write_csv(out_dir / "speaker_comparison.csv", rows)
    write_summary_json(out_dir / "dataset_summary.json", summary)

    # Charts
    print("Generating charts...")
    chart_dashboard(out_dir, original_summary, filtered_summary, rows)
    chart_total_records_speakers(out_dir, original_summary, filtered_summary)
    chart_train_val_split(out_dir, original_summary, filtered_summary)
    chart_speaker_distribution_sorted(out_dir, rows)
    chart_top_speakers(out_dir, rows, top_n=args.top_n)
    chart_removed_records(out_dir, rows, top_n=args.top_n)
    chart_retention_hist(out_dir, rows)
    chart_scatter_original_filtered(out_dir, rows)
    chart_distribution_hist(out_dir, rows)
    chart_pareto(out_dir, rows)
    chart_kept_dropped(out_dir, rows)
    chart_boxplot(out_dir, rows)
    chart_phoneme_vocab(out_dir, phoneme_vocab, top_n=min(args.top_n, 40))
    chart_filelist_lengths(out_dir, train_stats, val_stats, combined_stats)
    chart_cap_effect(out_dir, rows, filtered_map)
    chart_record_loss_waterfall(out_dir, original_summary, filtered_summary)

    # Print concise report
    print("")
    print("=" * 88)
    print("DONE")
    print("=" * 88)
    print(f"Original records : {fmt_int(orig_total)}")
    print(f"Filtered records : {fmt_int(filt_total)}")
    print(f"Removed records  : {fmt_int(removed_records)} ({fmt_pct(safe_div(removed_records, orig_total) * 100)})")
    print(f"Original speakers: {fmt_int(original_summary['speakers'])}")
    print(f"Filtered speakers: {fmt_int(filtered_summary['speakers'])}")
    print(f"Dropped speakers : {fmt_int(dropped_speakers)}")
    print("")
    print(f"Charts saved to  : {out_dir}")
    print(f"CSV report       : {out_dir / 'speaker_comparison.csv'}")
    print(f"JSON summary     : {out_dir / 'dataset_summary.json'}")
    print("=" * 88)

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate modern visualization charts for ViVoice dataset before/after filtering."
    )
    parser.add_argument(
        "--original-map",
        type=str,
        default=str(DEFAULT_ORIGINAL_MAP),
        help="Path tới speaker_id_map.json.backup_before_filter, đại diện data gốc trước lọc.",
    )
    parser.add_argument(
        "--filtered-map",
        type=str,
        default=str(DEFAULT_FILTERED_MAP),
        help="Path tới speaker_id_map.json sau khi lọc.",
    )
    parser.add_argument(
        "--train-list",
        type=str,
        default=str(DEFAULT_TRAIN_LIST),
        help="Path tới vivoice_train_list.txt sau lọc. Optional nhưng nên có để thống kê phoneme length.",
    )
    parser.add_argument(
        "--val-list",
        type=str,
        default=str(DEFAULT_VAL_LIST),
        help="Path tới vivoice_val_list.txt sau lọc. Optional nhưng nên có để thống kê phoneme length.",
    )
    parser.add_argument(
        "--phoneme-vocab",
        type=str,
        default=None,
        help="Path tới phoneme_vocab.json. Nếu không truyền, script thử tìm trong output/ và prepare_vivoice/.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(DEFAULT_OUT_DIR),
        help="Folder lưu PNG charts + CSV/JSON reports.",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default="|",
        help="Delimiter trong filelist. Mặc định: |",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=30,
        help="Số speaker/phoneme top để vẽ trong các chart top-N.",
    )

    args = parser.parse_args()

    try:
        code = run(args)
    except Exception as e:
        print("")
        print("ERROR:", e)
        print("")
        print("Gợi ý kiểm tra:")
        print("  1. Bạn có đang chạy script trong đúng folder prepare_vivoice không?")
        print("  2. Có tồn tại output/speaker_id_map.json và output/speaker_id_map.json.backup_before_filter không?")
        print("  3. Nếu phoneme_vocab.json nằm chỗ khác, hãy truyền --phoneme-vocab path.")
        print("  4. Nếu thiếu matplotlib/numpy: pip install matplotlib numpy")
        sys.exit(1)

    sys.exit(code)

if __name__ == "__main__":
    main()
