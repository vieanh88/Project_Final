"""
=============================================================
  ĐÁNH GIÁ THỰC NGHIỆM (Chương 5) — CER • RTF • render mẫu MOS/SMOS
=============================================================
Tự động hoá 3 trục đánh giá của báo cáo:

  1) render  : tổng hợp audio cho test set theo NHIỀU điều kiện mô hình
               (mô hình cuối vs mô hình TRƯỚC fine-tune) + sao chép bản ghi
               giọng thật, rồi sinh bộ mẫu ĐÃ ẨN DANH + đáp án cho Google Form
               (MOS độ tự nhiên + SMOS độ tương đồng giọng).
  2) rtf     : đo Real-Time Factor đúng chuẩn (warm-up + cuda.synchronize),
               so FP16 / FP32 / CPU, phân rã theo độ dài câu.
  3) cer     : cho audio qua ASR tiếng Việt (PhoWhisper-large + Whisper-large-v3),
               so với văn bản gốc -> Character Error Rate. Có tính "sàn CER"
               trên bản ghi giọng thật để tách lỗi TTS khỏi lỗi ASR.

Mọi đường dẫn engine/checkpoint/reference đọc từ inference/inference_config.yaml
(tái sử dụng config_loader + inference_engine — KHÔNG lặp lại logic).

------------------------------------------------------------
CÁCH CHẠY (từ thư mục TTS_StyleTTS2-lite-vi/):

  # B1. Render audio mọi điều kiện + tạo bộ mẫu cho Google Form
  python inference/evaluate_demo.py render

  # B2. Benchmark hiệu năng (RTF). Thêm --no-cpu để bỏ qua đo trên CPU (chậm)
  python inference/evaluate_demo.py rtf

  # B3. Tính CER (cần audio từ B1). Cần: pip install transformers jiwer
  python inference/evaluate_demo.py cer

  # Hoặc chạy tất cả:
  python inference/evaluate_demo.py all

Tham số phụ (đều có default hợp lý):
  --config         path inference_config.yaml
  --pre-checkpoint path checkpoint TRƯỚC fine-tune (mặc định kaggle_models/base_model_120k_vi.pth)
  --gt-wav-dir     thư mục chứa wav giọng thật
  --out-dir        thư mục xuất kết quả (mặc định output/eval)
  --asr            phowhisper | whisper | both  (mặc định both)
  --no-cpu         bỏ qua nhánh CPU khi đo RTF
  --repeats        số lần lặp mỗi câu khi đo RTF (mặc định 3, lấy trung vị)
=============================================================
"""

from __future__ import annotations

import sys as _sys

# Ép stdout/stderr sang UTF-8 — tránh UnicodeEncodeError khi print tiếng Việt/emoji
# trên Windows (console mặc định cp1252). Giống fix ở webui/backend/app.py.
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

import argparse
import csv
import gc
import json
import random
import shutil
import statistics
import sys
import time
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

# ----- Nạp code inference/ vào sys.path (giống pipeline.py) -----
INFERENCE_DIR = Path(__file__).resolve().parent          # inference/
PROJECT_ROOT = INFERENCE_DIR.parent                       # TTS_StyleTTS2-lite-vi/
if str(INFERENCE_DIR) not in sys.path:
    sys.path.insert(0, str(INFERENCE_DIR))

from config_loader import (                               # noqa: E402
    DEFAULT_CONFIG_PATH, load_config, cfg_value, resolve_path,
)

TESTSET_PATH = INFERENCE_DIR / "eval" / "testset.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "output" / "eval"
DEFAULT_PRE_CKPT = "kaggle_models/base_model_120k_vi.pth"   # checkpoint trước fine-tune (nếu có, để so sánh với model cuối)
DEFAULT_GT_WAV_DIR = "kaggle_upload/ngan-data-lite-vi/wavs"

# Tên hiển thị điều kiện (giữ trung lập, KHÔNG lộ tên thật/giọng cụ thể trong báo cáo)
COND_FINAL = "final"          # mô hình sau Giai đoạn 3 (epoch cuối)
COND_PRE = "pretrained"       # mô hình trước fine-tune giọng mục tiêu
COND_GT = "groundtruth"       # bản ghi giọng thật


# ============================================================
# Test set
# ============================================================
def load_testset() -> dict:
    if not TESTSET_PATH.exists():
        raise FileNotFoundError(f"Không thấy test set: {TESTSET_PATH}")
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not data.get("in_domain") and not data.get("out_domain"):
        raise ValueError("Test set rỗng (thiếu in_domain/out_domain).")
    return data


def all_sentences(ts: dict) -> list[dict]:
    """Gộp in_domain + out_domain thành 1 list thống nhất với khoá 'part'/'genre'."""
    out = []
    for s in ts.get("in_domain", []):
        out.append({"id": s["id"], "text": s["text"], "part": "in_domain",
                    "genre": "in_domain", "gt_wav": s.get("gt_wav")})
    for s in ts.get("out_domain", []):
        out.append({"id": s["id"], "text": s["text"], "part": "out_domain",
                    "genre": s.get("genre", "other"), "gt_wav": None})
    return out


# ============================================================
# CER — chuẩn hoá văn bản + Levenshtein ký tự
# ============================================================
_VN_UNITS = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]


def _vn_read_below_thousand(n: int, full: bool) -> str:
    """Đọc 0..999 theo tiếng Việt. full=True -> luôn đọc đủ 3 cấp (cho khối nghìn/triệu)."""
    tram, n = divmod(n, 100)
    chuc, donvi = divmod(n, 10)
    out = []
    if tram > 0 or full:
        out.append(_VN_UNITS[tram] + " trăm")
    if chuc == 0:
        if donvi > 0 and (tram > 0 or full):
            out.append("lẻ " + _VN_UNITS[donvi])
        elif donvi > 0:
            out.append(_VN_UNITS[donvi])
    elif chuc == 1:
        out.append("mười")
        if donvi == 5:
            out.append("lăm")
        elif donvi > 0:
            out.append(_VN_UNITS[donvi])
    else:
        out.append(_VN_UNITS[chuc] + " mươi")
        if donvi == 1:
            out.append("mốt")
        elif donvi == 5:
            out.append("lăm")
        elif donvi > 0:
            out.append(_VN_UNITS[donvi])
    return " ".join(out)


def vn_int_to_words(n: int) -> str:
    """Đọc số nguyên 0..(dưới 1 tỷ) thành chữ tiếng Việt (đủ cho dữ liệu test)."""
    if n == 0:
        return "không"
    parts = []
    ty, n = divmod(n, 1_000_000_000)
    trieu, n = divmod(n, 1_000_000)
    nghin, donvi = divmod(n, 1_000)
    if ty:
        parts.append(_vn_read_below_thousand(ty, False) + " tỷ")
    if trieu:
        parts.append(_vn_read_below_thousand(trieu, bool(ty)) + " triệu")
    if nghin:
        parts.append(_vn_read_below_thousand(nghin, bool(ty or trieu)) + " nghìn")
    if donvi:
        parts.append(_vn_read_below_thousand(donvi, bool(ty or trieu or nghin)))
    return " ".join(parts)


def normalize_for_cer(text: str) -> str:
    """
    Chuẩn hoá tiếng Việt trước khi tính CER:
      - NFC unicode, casefold (thường hoá)
      - bỏ mọi ký tự KHÔNG phải chữ/số/space (dấu câu, ngoặc kép...)
      - GIỮ dấu thanh tiếng Việt
      - ĐỔI mọi cụm chữ số -> chữ đọc tiếng Việt (vd "20" -> "hai mươi") để CER
        không bị phạt oan khi ASR xuất chữ số còn văn bản gốc viết bằng chữ.
      - gộp nhiều space -> 1, strip
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text).casefold()
    kept = []
    for ch in text:
        if ch.isalnum() or ch.isspace():
            kept.append(ch)
        else:
            kept.append(" ")  # dấu câu -> space để không dính chữ
    # Đổi token toàn chữ số -> chữ đọc
    tokens = "".join(kept).split()
    norm_tokens = []
    for tok in tokens:
        if tok.isdigit():
            try:
                norm_tokens.append(vn_int_to_words(int(tok)))
            except Exception:
                norm_tokens.append(tok)
        else:
            norm_tokens.append(tok)
    text = " ".join(norm_tokens)
    return unicodedata.normalize("NFC", text)


def char_error_rate(ref: str, hyp: str) -> tuple[float, int, int]:
    """
    CER = Levenshtein(ref, hyp) / len(ref)  (mức ký tự, đã chuẩn hoá).
    Trả về (cer, edit_distance, n_ref_chars).
    """
    r = normalize_for_cer(ref)
    h = normalize_for_cer(hyp)
    n = len(r)
    if n == 0:
        return (0.0 if len(h) == 0 else 1.0), len(h), 0
    # DP Levenshtein O(n*m), n,m nhỏ (vài trăm ký tự) -> đủ nhanh
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cost = 0 if rc == hc else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    dist = prev[-1]
    return dist / n, dist, n


# ============================================================
# Engine helper
# ============================================================
def build_engine(cfg: dict, checkpoint: Path, device: str, use_fp16: bool):
    """Khởi tạo StyleTTS2LiteVNInference với checkpoint/device/fp16 tuỳ biến."""
    from inference_engine import StyleTTS2LiteVNInference
    return StyleTTS2LiteVNInference(
        checkpoint_path=checkpoint,
        repo_root=resolve_path(cfg_value(cfg, "engine", "repo")),
        config_path=resolve_path(cfg_value(cfg, "engine", "model_config")),
        device=device,
        use_fp16=use_fp16,
    )


def free_engine(engine) -> None:
    import torch
    try:
        del engine
    except Exception:
        pass
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _resolve_conditions(cfg: dict, args) -> dict[str, Path]:
    """Map điều kiện -> checkpoint path (đã resolve)."""
    final_ckpt = resolve_path(cfg_value(cfg, "engine", "checkpoint"))
    pre_ckpt = resolve_path(args.pre_checkpoint or DEFAULT_PRE_CKPT)
    conds = {COND_FINAL: final_ckpt}
    if pre_ckpt and pre_ckpt.exists():
        conds[COND_PRE] = pre_ckpt
    else:
        print(f"  ⚠️  Bỏ qua điều kiện '{COND_PRE}': không thấy checkpoint {pre_ckpt}")
    return conds


# ============================================================
# CMD: render — tổng hợp audio mọi điều kiện + bộ mẫu Google Form
# ============================================================
def cmd_render(cfg: dict, args) -> None:
    SR = 24000
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    mos_dir = out_dir / "mos_samples"
    audio_dir.mkdir(parents=True, exist_ok=True)
    mos_dir.mkdir(parents=True, exist_ok=True)

    ts = load_testset()
    sents = all_sentences(ts)
    male_ref = resolve_path(cfg_value(cfg, "references", "male_ref"))
    conds = _resolve_conditions(cfg, args)
    denoise = float(cfg_value(cfg, "tts", "denoise"))
    split_dur = float(cfg_value(cfg, "tts", "split_dur"))

    durations: dict = {}   # (cond, id) -> audio_dur_sec

    # ----- Render từng điều kiện mô hình -----
    for cond, ckpt in conds.items():
        print(f"\n=== Render điều kiện '{cond}'  ({ckpt.name}) ===")
        engine = build_engine(cfg, ckpt, cfg_value(cfg, "engine", "device"),
                              bool(cfg_value(cfg, "engine", "use_fp16")))
        style = engine.compute_style(str(male_ref), denoise=denoise, split_dur=split_dur)
        cdir = audio_dir / cond
        cdir.mkdir(parents=True, exist_ok=True)
        for s in sents:
            try:
                wav = engine.synthesize(s["text"], style, speed=1.0)
            except Exception as e:                          # noqa: BLE001
                print(f"  [LỖI] {s['id']}: {type(e).__name__}: {e}")
                continue
            sf.write(str(cdir / f"{s['id']}.wav"), wav, SR)
            durations[f"{cond}|{s['id']}"] = round(len(wav) / SR, 3)
        free_engine(engine)

    # ----- Sao chép bản ghi giọng thật (groundtruth) cho in_domain -----
    gt_src_dir = resolve_path(args.gt_wav_dir or DEFAULT_GT_WAV_DIR)
    gt_dir = audio_dir / COND_GT
    gt_dir.mkdir(parents=True, exist_ok=True)
    gt_ok: set[str] = set()
    for s in sents:
        if s["part"] != "in_domain" or not s["gt_wav"]:
            continue
        src = gt_src_dir / s["gt_wav"] if gt_src_dir else None
        if src and src.exists():
            dst = gt_dir / f"{s['id']}.wav"
            shutil.copyfile(src, dst)
            gt_ok.add(s["id"])
            try:
                info = sf.info(str(dst))
                durations[f"{COND_GT}|{s['id']}"] = round(info.frames / info.samplerate, 3)
            except Exception:
                pass
        else:
            print(f"  ⚠️  Không thấy bản ghi thật cho {s['id']}: {src}")

    # ----- Dựng bộ mẫu ẨN DANH + đáp án cho Google Form -----
    # MOS/SMOS: mỗi (điều kiện, câu) thành 1 "stimulus". in_domain có đủ 3 điều kiện
    # (groundtruth/final/pretrained) -> dùng cho cả MOS + SMOS. out_domain có
    # final/pretrained -> dùng cho MOS độ tự nhiên theo thể loại.
    rng = random.Random(args.seed)
    stimuli = []
    for s in sents:
        ids_present = []
        for cond in (COND_GT, COND_FINAL, COND_PRE):
            if cond == COND_GT and s["id"] not in gt_ok:
                continue
            if cond in (COND_FINAL, COND_PRE) and cond not in conds:
                continue
            src = audio_dir / cond / f"{s['id']}.wav"
            if src.exists():
                ids_present.append((cond, src))
        for cond, src in ids_present:
            stimuli.append({"sentence_id": s["id"], "part": s["part"],
                            "genre": s["genre"], "condition": cond, "src": src,
                            "text": s["text"]})
    rng.shuffle(stimuli)

    manifest_rows = []
    for i, st in enumerate(stimuli, 1):
        anon = f"sample_{i:03d}.wav"
        shutil.copyfile(st["src"], mos_dir / anon)
        manifest_rows.append({
            "sample_file": anon,
            "true_condition": st["condition"],     # ĐÁP ÁN — giữ riêng, KHÔNG đưa người nghe
            "sentence_id": st["sentence_id"],
            "part": st["part"],
            "genre": st["genre"],
            "smos_eligible": "yes" if st["part"] == "in_domain" else "no",
            "text": st["text"],
        })

    # File reference cho SMOS (đoạn giọng mục tiêu để so sánh)
    if male_ref and male_ref.exists():
        shutil.copyfile(male_ref, mos_dir / "SMOS_reference.wav")

    key_csv = out_dir / "mos_manifest.csv"
    with open(key_csv, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)

    with open(out_dir / "render_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "conditions": list(conds.keys()) + ([COND_GT] if gt_ok else []),
            "n_sentences": len(sents),
            "n_in_domain": len(ts.get("in_domain", [])),
            "n_out_domain": len(ts.get("out_domain", [])),
            "n_mos_stimuli": len(stimuli),
            "durations_sec": durations,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Render xong.")
    print(f"   Audio theo điều kiện : {audio_dir}")
    print(f"   Bộ mẫu ẩn danh (Form): {mos_dir}  ({len(stimuli)} mẫu)")
    print(f"   ĐÁP ÁN (giữ riêng)   : {key_csv}")
    print(f"   -> Dùng inference/eval/testset.json + mos_manifest.csv để dựng Google Form.")


# ============================================================
# CMD: rtf — benchmark hiệu năng
# ============================================================
def cmd_rtf(cfg: dict, args) -> None:
    import torch
    SR = 24000
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ts = load_testset()
    sents = all_sentences(ts)
    male_ref = resolve_path(cfg_value(cfg, "references", "male_ref"))
    final_ckpt = resolve_path(cfg_value(cfg, "engine", "checkpoint"))
    denoise = float(cfg_value(cfg, "tts", "denoise"))
    split_dur = float(cfg_value(cfg, "tts", "split_dur"))

    # Các cấu hình phần cứng/độ chính xác cần đo
    bench_cfgs = []
    if torch.cuda.is_available():
        bench_cfgs.append(("cuda_fp16", "cuda", True))
        bench_cfgs.append(("cuda_fp32", "cuda", False))
    if not args.no_cpu:
        bench_cfgs.append(("cpu_fp32", "cpu", False))
    if not bench_cfgs:
        bench_cfgs.append(("cpu_fp32", "cpu", False))

    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    rows = []
    summary = {"gpu": gpu_name, "repeats": args.repeats, "configs": {}}

    for tag, device, fp16 in bench_cfgs:
        # CPU rất chậm -> chỉ đo trên tập con để tiết kiệm thời gian
        bench_sents = sents if device != "cpu" else sents[: args.cpu_subset]
        print(f"\n=== RTF [{tag}]  ({len(bench_sents)} câu × {args.repeats} lần) ===")
        engine = build_engine(cfg, final_ckpt, device, fp16)
        style = engine.compute_style(str(male_ref), denoise=denoise, split_dur=split_dur)

        # Warm-up (bỏ — gồm CUDA init / kernel compile)
        for _ in range(2):
            engine.synthesize(bench_sents[0]["text"], style, speed=1.0)
        if device == "cuda":
            torch.cuda.synchronize()

        per_cfg_rtf = []
        for s in bench_sents:
            times = []
            audio_dur = None
            for _ in range(args.repeats):
                if device == "cuda":
                    torch.cuda.synchronize()
                t0 = time.perf_counter()
                wav = engine.synthesize(s["text"], style, speed=1.0)
                if device == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - t0)
                audio_dur = len(wav) / SR
            t_med = statistics.median(times)
            rtf = t_med / audio_dur if audio_dur else float("nan")
            per_cfg_rtf.append(rtf)
            rows.append({
                "config": tag, "sentence_id": s["id"], "genre": s["genre"],
                "n_chars": len(s["text"]), "audio_sec": round(audio_dur, 3),
                "synth_sec_median": round(t_med, 4), "rtf": round(rtf, 4),
            })
        summary["configs"][tag] = {
            "n_sentences": len(bench_sents),
            "rtf_mean": round(statistics.mean(per_cfg_rtf), 4),
            "rtf_std": round(statistics.pstdev(per_cfg_rtf), 4) if len(per_cfg_rtf) > 1 else 0.0,
            "rtf_min": round(min(per_cfg_rtf), 4),
            "rtf_max": round(max(per_cfg_rtf), 4),
        }
        print(f"   RTF trung bình = {summary['configs'][tag]['rtf_mean']:.4f} "
              f"± {summary['configs'][tag]['rtf_std']:.4f}")
        free_engine(engine)

    with open(out_dir / "rtf_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(out_dir / "rtf_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ RTF xong. Bảng tổng hợp:")
    for tag, st in summary["configs"].items():
        print(f"   {tag:12s}: RTF = {st['rtf_mean']:.4f} ± {st['rtf_std']:.4f} "
              f"(min {st['rtf_min']}, max {st['rtf_max']})")
    print(f"   Chi tiết: {out_dir/'rtf_results.csv'}")


# ============================================================
# CMD: cer — ASR (PhoWhisper + Whisper) -> CER
# ============================================================
def _load_asr(name: str):
    """Trả về callable(wav_path)->str. Lazy import transformers."""
    import torch
    from transformers import pipeline
    model_id = {
        "phowhisper": "vinai/PhoWhisper-large",
        "whisper": "openai/whisper-large-v3",
    }[name]
    device = 0 if torch.cuda.is_available() else -1
    print(f"  Loading ASR '{name}' = {model_id} (device={'cuda' if device == 0 else 'cpu'})…")
    asr = pipeline(
        "automatic-speech-recognition", model=model_id,
        device=device, chunk_length_s=30,
        torch_dtype=torch.float16 if device == 0 else torch.float32,
    )
    gen_kwargs = {"language": "vi", "task": "transcribe"}

    def transcribe(wav_path: str) -> str:
        out = asr(wav_path, generate_kwargs=gen_kwargs)
        return (out.get("text") or "").strip()

    return transcribe, asr


def cmd_cer(cfg: dict, args) -> None:
    import torch
    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Chưa có audio để chấm CER ({audio_dir}). Chạy 'render' trước."
        )

    ts = load_testset()
    sents = all_sentences(ts)
    text_by_id = {s["id"]: s for s in sents}

    # Liệt kê (condition, id, wav, ref_text)
    items = []
    for cond in (COND_FINAL, COND_PRE, COND_GT):
        cdir = audio_dir / cond
        if not cdir.exists():
            continue
        for wav in sorted(cdir.glob("*.wav")):
            sid = wav.stem
            s = text_by_id.get(sid)
            if s is None:
                continue
            items.append({"condition": cond, "sentence_id": sid,
                          "genre": s["genre"], "part": s["part"],
                          "wav": str(wav), "ref": s["text"]})
    if not items:
        raise RuntimeError("Không thấy file audio nào khớp test set.")

    backends = {"phowhisper": "phowhisper", "whisper": "whisper"}
    if args.asr == "phowhisper":
        backends = {"phowhisper": "phowhisper"}
    elif args.asr == "whisper":
        backends = {"whisper": "whisper"}

    rows = []
    for bname in backends:
        try:
            transcribe, asr_obj = _load_asr(bname)
        except Exception as e:                              # noqa: BLE001
            print(f"  ⚠️  Bỏ qua ASR '{bname}': {type(e).__name__}: {e}")
            continue
        print(f"  Chạy {bname} trên {len(items)} file…")
        for it in items:
            try:
                hyp = transcribe(it["wav"])
            except Exception as e:                          # noqa: BLE001
                print(f"    [LỖI] {it['sentence_id']} ({it['condition']}): {e}")
                continue
            cer, dist, n = char_error_rate(it["ref"], hyp)
            rows.append({
                "asr": bname, "condition": it["condition"],
                "sentence_id": it["sentence_id"], "genre": it["genre"],
                "part": it["part"], "n_ref_chars": n, "edit_dist": dist,
                "cer": round(cer, 4), "hyp": hyp,
            })
        del transcribe, asr_obj
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("Không tính được CER (ASR đều fail?). Kiểm tra transformers.")

    with open(out_dir / "cer_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Tổng hợp CER theo (asr, condition) và (asr, condition, genre)
    def agg(key_fn) -> dict:
        buckets: dict = {}
        for r in rows:
            buckets.setdefault(key_fn(r), []).append(r["cer"])
        return {k: {"cer_mean": round(statistics.mean(v), 4),
                    "cer_std": round(statistics.pstdev(v), 4) if len(v) > 1 else 0.0,
                    "n": len(v)} for k, v in buckets.items()}

    summary = {
        "by_condition": agg(lambda r: f"{r['asr']}|{r['condition']}"),
        "by_genre": agg(lambda r: f"{r['asr']}|{r['condition']}|{r['genre']}"),
    }
    with open(out_dir / "cer_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n✅ CER xong. Tổng hợp theo điều kiện:")
    for k, v in sorted(summary["by_condition"].items()):
        print(f"   {k:28s}: CER = {v['cer_mean']*100:5.2f}%  (n={v['n']})")
    print(f"   Chi tiết: {out_dir/'cer_results.csv'}")
    print(f"   Lưu ý: CER trên '{COND_GT}' = SÀN của ASR (lỗi không phải do TTS).")


# ============================================================
# CMD: secs — độ tương đồng giọng KHÁCH QUAN (Speaker Encoder Cosine Similarity)
# ============================================================
def cmd_secs(cfg: dict, args) -> None:
    """
    SECS = cosine similarity giữa embedding giọng của audio tổng hợp và một
    'tâm giọng mục tiêu' (trung bình embedding các bản ghi giọng thật in-domain).
    Dùng mô hình speaker-verification WavLM (microsoft/wavlm-base-plus-sv) —
    KHÔNG cần cài thêm thư viện (đã có transformers).
    """
    import torch
    import librosa
    import numpy as np
    from transformers import AutoFeatureExtractor, WavLMForXVector

    out_dir = Path(args.out_dir)
    audio_dir = out_dir / "audio"
    gt_dir = audio_dir / COND_GT
    if not gt_dir.exists() or not any(gt_dir.glob("*.wav")):
        raise FileNotFoundError(
            f"Cần audio giọng thật ở {gt_dir} làm tham chiếu SECS. Chạy 'render' trước."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "microsoft/wavlm-base-plus-sv"
    print(f"  Loading speaker encoder {model_id} (device={device})…")
    fe = AutoFeatureExtractor.from_pretrained(model_id)
    model = WavLMForXVector.from_pretrained(model_id).to(device).eval()

    @torch.no_grad()
    def embed(wav_path: str) -> np.ndarray:
        wav, _ = librosa.load(wav_path, sr=16000, mono=True)
        inp = fe(wav, sampling_rate=16000, return_tensors="pt", padding=True)
        emb = model(**{k: v.to(device) for k, v in inp.items()}).embeddings
        emb = torch.nn.functional.normalize(emb, dim=-1)
        return emb.squeeze(0).cpu().numpy()

    def cos(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))  # đã chuẩn hoá L2

    # Tâm giọng mục tiêu = trung bình embedding các bản ghi thật
    gt_files = sorted(gt_dir.glob("*.wav"))
    gt_embs = {p.stem: embed(str(p)) for p in gt_files}
    centroid = np.mean(np.stack(list(gt_embs.values())), axis=0)
    centroid = centroid / (np.linalg.norm(centroid) + 1e-9)

    rows = []
    # Trần SECS: mỗi bản ghi thật so với tâm của CÁC bản ghi thật còn lại (leave-one-out)
    for sid, e in gt_embs.items():
        others = [v for k, v in gt_embs.items() if k != sid]
        if not others:
            continue
        c = np.mean(np.stack(others), axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        rows.append({"condition": COND_GT, "sentence_id": sid, "secs": round(cos(e, c), 4)})

    # SECS của audio tổng hợp (final / pretrained) so với tâm giọng mục tiêu
    for cond in (COND_FINAL, COND_PRE):
        cdir = audio_dir / cond
        if not cdir.exists():
            continue
        for p in sorted(cdir.glob("*.wav")):
            rows.append({"condition": cond, "sentence_id": p.stem,
                         "secs": round(cos(embed(str(p)), centroid), 4)})

    with open(out_dir / "secs_results.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["condition", "sentence_id", "secs"])
        w.writeheader()
        w.writerows(rows)

    summary = {}
    for cond in (COND_GT, COND_FINAL, COND_PRE):
        vals = [r["secs"] for r in rows if r["condition"] == cond]
        if vals:
            summary[cond] = {"secs_mean": round(statistics.mean(vals), 4),
                             "secs_std": round(statistics.pstdev(vals), 4) if len(vals) > 1 else 0.0,
                             "n": len(vals)}
    with open(out_dir / "secs_summary.json", "w", encoding="utf-8") as f:
        json.dump({"model": model_id, "by_condition": summary}, f, ensure_ascii=False, indent=2)

    print("\n✅ SECS xong. Tổng hợp (cosine, càng cao càng giống giọng mục tiêu):")
    for cond in (COND_GT, COND_FINAL, COND_PRE):
        if cond in summary:
            s = summary[cond]
            print(f"   {cond:12s}: SECS = {s['secs_mean']:.4f} ± {s['secs_std']:.4f}  (n={s['n']})")
    print(f"   Chi tiết: {out_dir/'secs_results.csv'}")
    print(f"   Lưu ý: '{COND_GT}' = trần (giọng thật vs giọng thật, leave-one-out).")


# ============================================================
# Main
# ============================================================
def main() -> None:
    ap = argparse.ArgumentParser(description="Đánh giá thực nghiệm StyleTTS2-lite-vi (Chương 5)")
    ap.add_argument("task", choices=["render", "rtf", "cer", "secs", "all"],
                    help="render | rtf | cer | secs | all")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--pre-checkpoint", default=None,
                    help=f"checkpoint trước fine-tune (mặc định {DEFAULT_PRE_CKPT})")
    ap.add_argument("--gt-wav-dir", default=None,
                    help=f"thư mục wav giọng thật (mặc định {DEFAULT_GT_WAV_DIR})")
    ap.add_argument("--asr", choices=["phowhisper", "whisper", "both"], default="both")
    ap.add_argument("--no-cpu", action="store_true", help="bỏ qua đo RTF trên CPU")
    ap.add_argument("--cpu-subset", type=int, default=5, help="số câu đo RTF trên CPU")
    ap.add_argument("--repeats", type=int, default=3, help="số lần lặp mỗi câu khi đo RTF")
    ap.add_argument("--seed", type=int, default=42, help="seed trộn mẫu MOS")
    args = ap.parse_args()

    cfg = load_config(args.config)

    if args.task in ("render", "all"):
        cmd_render(cfg, args)
    if args.task in ("rtf", "all"):
        cmd_rtf(cfg, args)
    if args.task in ("cer", "all"):
        cmd_cer(cfg, args)
    if args.task in ("secs", "all"):
        cmd_secs(cfg, args)


if __name__ == "__main__":
    main()
