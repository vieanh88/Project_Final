"""
=============================================================
  D1: VERIFY FEMALE REFERENCE AUDIO
=============================================================
Mục đích:
  Bạn đã cung cấp 1 file .wav giọng NỮ (cho role 'character_female').
  Script này kiểm tra file đó có dùng được cho inference không, bằng cách:

    1. HEALTH CHECK    — verify format, duration, sample rate, RMS energy
    2. STYLE EXTRACT   — gọi engine.compute_style() → check tensor valid
    3. SYNTHESIZE TEST — render 3 câu đa dạng với giọng nữ này
                         + (optional) so sánh với giọng Ngạn (--male-ref)

Tên file vốn là 'download_female_ref' theo roadmap ban đầu (định download
từ ViVoice). Bạn đã chọn option "tự cung cấp", nên file thực chất là
VERIFY script. Giữ tên cũ cho khớp roadmap.

Cách dùng (đường dẫn ref + engine đọc từ inference_config.yaml, section `references:`/`engine:`):

    # Test female (+ male nếu references.male_ref có) — load model + render
    python inference/download_female_ref.py --config inference/inference_config.yaml

    # Chỉ health check (nhanh, không load model)
    python inference/download_female_ref.py --config inference/inference_config.yaml --skip-synthesize

    # Override nhanh file female qua CLI (ưu tiên CLI > config > default)
    python inference/download_female_ref.py --config inference/inference_config.yaml \\
        --female-ref D:/path/khac/female.wav

Output: 3-6 file .wav trong output/female_ref_test/
=============================================================
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

# Tham số & đường dẫn tập trung trong inference_config.yaml (xem config_loader.py)
from config_loader import (
    SCRIPT_DIR,
    PROJECT_ROOT,
    DEFAULT_CONFIG_PATH,
    load_config,
    cfg_value,
    engine_kwargs,
    reference_path,
)

OUTPUT_DIR = PROJECT_ROOT / "output" / "female_ref_test"


# 3 câu test đa dạng — kể chuyện / hội thoại nữ / cảm thán
TEST_SENTENCES = [
    {
        "id": "narration",
        "label": "Kể chuyện (trung tính)",
        "text": "Đêm đó, cô gái trẻ lặng lẽ bước ra khỏi căn nhà cũ kỹ.",
    },
    {
        "id": "dialogue",
        "label": "Lời thoại nữ (hỏi)",
        "text": "Anh ơi, anh có nghe thấy tiếng gì lạ ngoài hành lang không?",
    },
    {
        "id": "exclamation",
        "label": "Cảm thán (sợ hãi)",
        "text": "Trời ơi, ai vừa gọi tên tôi vậy? Lạnh quá!",
    },
]


# ============================================================
# 1. HEALTH CHECK
# ============================================================
def health_check(audio_path: Path) -> dict:
    """
    Kiểm tra cơ bản file audio. Không cần model.

    Returns:
        dict với các fields chẩn đoán + warnings list.
    """
    import librosa

    result = {
        "path": str(audio_path),
        "exists": audio_path.exists(),
        "warnings": [],
        "errors": [],
        "ok": True,
    }

    if not audio_path.exists():
        result["errors"].append(f"File không tồn tại: {audio_path}")
        result["ok"] = False
        return result

    # File size
    size_mb = audio_path.stat().st_size / 1e6
    result["size_mb"] = round(size_mb, 2)

    # Load với soundfile để lấy metadata gốc
    try:
        info = sf.info(str(audio_path))
        result["format"] = info.format
        result["original_sr"] = info.samplerate
        result["channels"] = info.channels
        result["duration_sec"] = round(info.frames / info.samplerate, 3)
    except Exception as e:
        result["errors"].append(f"Đọc metadata fail: {e}")
        result["ok"] = False
        return result

    # Warnings về format
    if info.samplerate < 16000:
        result["warnings"].append(
            f"Sample rate thấp ({info.samplerate} Hz). Khuyến nghị ≥ 22050 Hz "
            "để compute_style ra style vector chất lượng."
        )
    if info.channels > 1:
        result["warnings"].append(
            f"Audio có {info.channels} channels (stereo/multi). "
            "Engine sẽ tự convert mono — OK nhưng có thể mất ngữ điệu nếu 2 channel khác nhau."
        )
    if info.frames / info.samplerate < 3.0:
        result["warnings"].append(
            f"Duration ngắn ({info.frames / info.samplerate:.1f}s). "
            "Khuyến nghị 5-15s để compute_style chia thành đoạn 2s + lấy trung bình."
        )
    if info.frames / info.samplerate > 60.0:
        result["warnings"].append(
            f"Duration dài ({info.frames / info.samplerate:.1f}s). "
            "Engine sẽ cắt còn 20s đầu. Tốt nhất < 30s."
        )

    # Load thật về 24kHz mono để check RMS + clipping
    try:
        wave, _ = librosa.load(str(audio_path), sr=24000, mono=True)
    except Exception as e:
        result["errors"].append(f"Librosa decode fail: {e}")
        result["ok"] = False
        return result

    # RMS energy (mức to nhỏ)
    rms = float(np.sqrt(np.mean(wave ** 2)))
    peak = float(np.max(np.abs(wave)))
    result["rms"] = round(rms, 4)
    result["peak"] = round(peak, 4)
    result["len_after_resample"] = len(wave)

    if rms < 0.005:
        result["warnings"].append(
            f"Audio rất nhỏ tiếng (RMS={rms:.4f}). Có thể chỉ là silence hoặc "
            "speech rất nhẹ — compute_style có thể không stable."
        )
    if peak >= 0.99:
        result["warnings"].append(
            f"Audio bị clipping (peak={peak:.3f} ≥ 0.99). Có thể méo style."
        )

    # Check duration SAU khi trim silence (engine sẽ làm điều này)
    wave_trimmed, _ = librosa.effects.trim(wave, top_db=30)
    dur_trimmed = len(wave_trimmed) / 24000
    result["dur_after_trim"] = round(dur_trimmed, 3)

    if dur_trimmed < 0.5:
        result["errors"].append(
            f"Sau khi trim silence chỉ còn {dur_trimmed:.2f}s — engine sẽ raise ValueError. "
            "Hãy cung cấp file có speech rõ ràng > 1s."
        )
        result["ok"] = False
    elif dur_trimmed < 3.0:
        result["warnings"].append(
            f"Sau trim chỉ còn {dur_trimmed:.2f}s — không thể chia đoạn 2s "
            "để lấy style trung bình. Style sẽ tính từ full audio."
        )

    return result


def print_health_report(report: dict, name: str = "Audio") -> None:
    """In báo cáo health check đẹp."""
    print(f"\n{'=' * 60}")
    print(f"HEALTH CHECK — {name}")
    print(f"{'=' * 60}")
    print(f"  Path        : {report['path']}")

    if "size_mb" in report:
        print(f"  Size        : {report['size_mb']} MB")
    if "format" in report:
        print(f"  Format      : {report['format']}")
        print(f"  Sample rate : {report['original_sr']} Hz")
        print(f"  Channels    : {report['channels']}")
        print(f"  Duration    : {report['duration_sec']}s")
    if "rms" in report:
        print(f"  RMS         : {report['rms']}")
        print(f"  Peak        : {report['peak']}")
        print(f"  Dur (trim)  : {report['dur_after_trim']}s")

    if report["warnings"]:
        print(f"\n  ⚠️  Warnings ({len(report['warnings'])}):")
        for w in report["warnings"]:
            print(f"    - {w}")

    if report["errors"]:
        print(f"\n  ❌ Errors ({len(report['errors'])}):")
        for e in report["errors"]:
            print(f"    - {e}")

    if report["ok"] and not report["warnings"]:
        print(f"\n  ✅ Health check PASS — file ready to use")
    elif report["ok"]:
        print(f"\n  ⚠️  Health check PASS với warnings — vẫn dùng được, nhưng review warnings")
    else:
        print(f"\n  ❌ Health check FAIL — file không dùng được")


# ============================================================
# 2 + 3. STYLE EXTRACT + SYNTHESIZE TEST
# ============================================================
def run_full_test(
    engine,
    ref_path: Path,
    role_name: str,
    output_subdir: Path,
) -> dict:
    """
    Trích style + render 3 câu test.

    Args:
        engine:        StyleTTS2LiteVNInference instance
        ref_path:      file .wav reference
        role_name:     'female' hoặc 'male' (dùng đặt tên file output)
        output_subdir: nơi save .wav output

    Returns:
        dict với style_shape + list paths đã render
    """
    output_subdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"STYLE EXTRACT + SYNTHESIZE TEST — role={role_name}")
    print(f"{'=' * 60}")

    # Compute style
    print(f"\n[1/4] Computing style từ: {ref_path.name} ...")
    try:
        style = engine.compute_style(str(ref_path), denoise=0.3, split_dur=2.0)
    except Exception as e:
        print(f"  ❌ compute_style FAIL: {type(e).__name__}: {e}")
        return {"ok": False, "error": str(e)}

    # Verify tensor
    if torch.isnan(style).any():
        print(f"  ❌ Style chứa NaN values!")
        return {"ok": False, "error": "Style NaN"}
    if torch.isinf(style).any():
        print(f"  ❌ Style chứa Inf values!")
        return {"ok": False, "error": "Style Inf"}

    print(f"  ✅ Style shape: {tuple(style.shape)}")
    print(f"     Mean: {style.mean().item():+.4f}  Std: {style.std().item():.4f}  "
          f"Min: {style.min().item():+.4f}  Max: {style.max().item():+.4f}")

    # Render 3 câu test
    rendered = []
    for idx, sent in enumerate(TEST_SENTENCES, start=1):
        print(f"\n[{idx+1}/4] {sent['label']}")
        print(f"  Text: {sent['text']}")
        try:
            phoneme = engine.text_to_phoneme(sent['text'])
            print(f"  Phn : {phoneme[:80]}{'...' if len(phoneme) > 80 else ''}")

            import time
            t0 = time.time()
            wav = engine.synthesize(phoneme, style, already_phonemized=True)
            elapsed = time.time() - t0

            out_path = output_subdir / f"{role_name}_{sent['id']}.wav"
            sf.write(str(out_path), wav, 24000)

            dur = len(wav) / 24000
            rtf = elapsed / dur
            print(f"  ✅ Saved: {out_path.name}  ({dur:.2f}s, RTF={rtf:.3f}x)")
            rendered.append({
                "id": sent["id"],
                "label": sent["label"],
                "path": str(out_path),
                "duration": round(dur, 2),
                "rtf": round(rtf, 3),
            })
        except Exception as e:
            print(f"  ❌ Synthesize FAIL: {type(e).__name__}: {e}")
            rendered.append({"id": sent["id"], "error": str(e)})

    return {
        "ok": True,
        "style_shape": tuple(style.shape),
        "rendered": rendered,
    }


# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    # File config YAML — nguồn chính của mọi đường dẫn & tham số
    parser.add_argument(
        "--config", type=str, default=str(DEFAULT_CONFIG_PATH),
        help=f"Path tới inference config YAML (default: {DEFAULT_CONFIG_PATH.name})",
    )
    # ---- Override CLI (None = không truyền -> dùng config -> default) ----
    parser.add_argument("--female-ref", type=str, default=None,
                        help="[references.female_ref] file .wav giọng nữ")
    parser.add_argument("--male-ref", type=str, default=None,
                        help="[references.male_ref] file .wav giọng Ngạn để so sánh")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[engine.checkpoint] path tới epoch_*.pth")
    parser.add_argument("--repo", type=str, default=None,
                        help="[engine.repo] folder StyleTTS2-lite")
    parser.add_argument("--model-config", type=str, default=None,
                        help="[engine.model_config] config.yaml KIẾN TRÚC model (khác --config)")
    parser.add_argument("--device", type=str, default=None,
                        help="[engine.device] auto | cuda | cpu")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="[engine.use_fp16] --fp16 / --no-fp16")
    parser.add_argument("--skip-synthesize", action=argparse.BooleanOptionalAction, default=None,
                        help="[verify_ref.skip_synthesize] chỉ health check, không load model")
    args = parser.parse_args()

    print("=" * 60)
    print("D1 — VERIFY FEMALE REFERENCE AUDIO")
    print("=" * 60)

    # ===== Resolve config (CLI > YAML > default) =====
    cfg = load_config(args.config)

    female_path = reference_path(cfg, args, "female_ref")
    male_path = reference_path(cfg, args, "male_ref")
    skip_synth = cfg_value(cfg, "verify_ref", "skip_synthesize", args.skip_synthesize)
    eng_kwargs = engine_kwargs(cfg, args)

    print(f"Project root  : {PROJECT_ROOT}")
    print(f"Female ref    : {female_path}")
    print(f"Male ref      : {male_path or '(skip)'}")
    print(f"Skip synth    : {skip_synth}")

    # ===== Step 1: Health check =====
    female_report = health_check(female_path)
    print_health_report(female_report, name="FEMALE")

    male_report = None
    if male_path:
        male_report = health_check(male_path)
        print_health_report(male_report, name="MALE (Ngạn)")

    if not female_report["ok"]:
        print(f"\n❌ Female ref FAIL health check. Hãy fix file trước khi tiếp tục.")
        print(f"\nGợi ý sửa lỗi:")
        print(f"  - File quá ngắn  → record/cắt lại đoạn 5-15s")
        print(f"  - File silence   → check audio có speech thật không")
        print(f"  - Decode fail    → convert sang .wav 16-bit PCM bằng audacity/ffmpeg:")
        print(f"      ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 female_ref.wav")
        sys.exit(1)

    if skip_synth:
        print(f"\n✅ Health check xong (skip synthesize theo flag).")
        sys.exit(0)

    # ===== Step 2: Load engine =====
    print(f"\n{'=' * 60}")
    print(f"LOADING INFERENCE ENGINE")
    print(f"{'=' * 60}")

    sys.path.insert(0, str(SCRIPT_DIR))
    try:
        from inference_engine import StyleTTS2LiteVNInference
    except ImportError as e:
        print(f"❌ Không import được inference_engine: {e}")
        print(f"   Đảm bảo file inference_engine.py nằm cùng folder: {SCRIPT_DIR}")
        sys.exit(1)

    try:
        engine = StyleTTS2LiteVNInference(**eng_kwargs)
    except Exception as e:
        print(f"❌ Engine init FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

    # ===== Step 3: Test female ref =====
    female_result = run_full_test(
        engine, female_path, role_name="female",
        output_subdir=OUTPUT_DIR,
    )

    # ===== Step 4 (optional): Test male ref =====
    male_result = None
    if male_path and male_report["ok"]:
        male_result = run_full_test(
            engine, male_path, role_name="male_ngan",
            output_subdir=OUTPUT_DIR,
        )

    # ===== Tổng kết =====
    print(f"\n{'=' * 60}")
    print(f"TỔNG KẾT")
    print(f"{'=' * 60}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"\nFEMALE:")
    if female_result["ok"]:
        print(f"  ✅ Pass. {len(female_result['rendered'])} sample đã render.")
    else:
        print(f"  ❌ Fail: {female_result.get('error', '?')}")
    if male_result:
        print(f"\nMALE (Ngạn):")
        if male_result["ok"]:
            print(f"  ✅ Pass. {len(male_result['rendered'])} sample đã render.")
        else:
            print(f"  ❌ Fail: {male_result.get('error', '?')}")

    print(f"\n👉 Hành động tiếp theo:")
    print(f"  1. Mở folder {OUTPUT_DIR} để NGHE 3-6 file .wav")
    print(f"  2. Đánh giá theo bảng:")
    print(f"     ┌─────────────────────┬──────────────────────────────┐")
    print(f"     │ Tiêu chí            │ Tốt → đi tiếp D2/D3          │")
    print(f"     ├─────────────────────┼──────────────────────────────┤")
    print(f"     │ Giọng nữ rõ ràng    │ KHÔNG bị méo, KHÔNG giả nam  │")
    print(f"     │ Phát âm tiếng Việt  │ Rõ, không lắp, không nuốt từ │")
    print(f"     │ Tự nhiên            │ Không robot, không monotone  │")
    print(f"     └─────────────────────┴──────────────────────────────┘")
    print(f"  3. Nếu giọng nữ KHÔNG đạt → cung cấp file .wav khác và chạy lại D1")
    print(f"  4. Nếu OK → đi tiếp D2 (nlp_generator.py với Gemini)")

if __name__ == "__main__":
    main()