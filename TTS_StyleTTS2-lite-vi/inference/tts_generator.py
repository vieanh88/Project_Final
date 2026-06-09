"""
=============================================================
  D3: TTS GENERATOR — Audiobook synthesis (Phase 2)
=============================================================
Mục đích:
  File CUỐI CÙNG của backend pipeline. Đọc script.json từ D2 +
  load D0 inference engine + 2 reference audio (Ngạn + giọng nữ),
  loop sinh audio cho từng câu, chèn silence padding theo
  pause_after_ms, concat thành audiobook hoàn chỉnh.

Quy ước role -> style:
    narrator         -> giọng Ngạn (male_ref)
    character_male   -> giọng Ngạn (cùng, vì Ngạn là giọng chính)
    character_female -> giọng nữ (female_ref)

Logic theo tài liệu thiết kế gốc (PDF Phần 2 Bước 3):
    audio_chunks = []
    for line in script:
        wav = engine.synthesize(line.text, style[line.role])
        audio_chunks.append(wav)
        silence = np.zeros(int(line.pause_after_ms / 1000 * 24000))
        audio_chunks.append(silence)
    final_wav = np.concatenate(audio_chunks)
    sf.write(out, final_wav, 24000)

Cách dùng (mọi đường dẫn & tham số đọc từ inference_config.yaml):
    python inference/tts_generator.py --config inference/inference_config.yaml

Override nhanh qua CLI (đè giá trị trong config — ưu tiên CLI > config > default):
    python inference/tts_generator.py --config inference/inference_config.yaml \\
        --script output/nlp/ghost1.json --output output/audiobooks/ghost1.wav \\
        --narrator-speed 0.9

Các tham số chỉnh trong section `tts:` / `references:` / `engine:` của config:
    script, output, male_ref, female_ref, checkpoint, narrator_speed,
    character_speed, save_lines, skip_on_error, normalize, denoise, split_dur
=============================================================
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm

# Tham số & đường dẫn tập trung trong inference_config.yaml (xem config_loader.py)
from config_loader import (
    SCRIPT_DIR,
    DEFAULT_CONFIG_PATH,
    load_config,
    cfg_value,
    engine_kwargs,
    reference_path,
    resolve_path,
)

SR = 24000  # Sample rate cố định của lite-vi


# ============================================================
# 1. AUDIOBOOK SYNTHESIZER CLASS
# ============================================================
class AudiobookSynthesizer:
    """
    Wrap quanh inference engine để render full script.json -> audiobook.

    Init 1 lần: load engine + compute 2 styles (male/Ngạn + female).
    Sau đó synthesize_script() có thể gọi nhiều lần với script khác nhau.
    """

    # Mapping role -> reference type
    ROLE_TO_REF_TYPE = {
        "narrator": "male",          # giọng Ngạn
        "character_male": "male",    # giọng Ngạn
        "character_female": "female",
    }

    def __init__(
        self,
        engine,
        male_ref_path: Path,
        female_ref_path: Path,
        denoise: float = 0.3,
        split_dur: float = 2.0,
    ):
        """
        Args:
            engine:           StyleTTS2LiteVNInference instance
            male_ref_path:    file .wav giọng Ngạn (cho narrator + character_male)
            female_ref_path:  file .wav giọng nữ (cho character_female)
            denoise:          % noisereduce blend khi compute_style (0-1)
            split_dur:        chia audio reference thành đoạn N giây để avg style
        """
        self.engine = engine
        self.male_ref = Path(male_ref_path).resolve()
        self.female_ref = Path(female_ref_path).resolve()

        for p, name in [(self.male_ref, "male_ref"), (self.female_ref, "female_ref")]:
            if not p.exists():
                raise FileNotFoundError(f"{name} không tồn tại: {p}")

        # Compute styles 1 lần — cache trong self.styles
        print(f"  Computing male style from: {self.male_ref.name}...")
        t0 = time.time()
        male_style = engine.compute_style(
            str(self.male_ref), denoise=denoise, split_dur=split_dur
        )
        print(f"    shape={tuple(male_style.shape)}, time={time.time()-t0:.2f}s")

        print(f"  Computing female style from: {self.female_ref.name}...")
        t0 = time.time()
        female_style = engine.compute_style(
            str(self.female_ref), denoise=denoise, split_dur=split_dur
        )
        print(f"    shape={tuple(female_style.shape)}, time={time.time()-t0:.2f}s")

        # Mapping role -> style tensor
        self.styles: Dict[str, torch.Tensor] = {
            "narrator": male_style,
            "character_male": male_style,
            "character_female": female_style,
        }

    def synthesize_line(
        self,
        text: str,
        role: str,
        speed: float = 1.0,
    ) -> np.ndarray:
        """Synthesize 1 line audio."""
        if role not in self.styles:
            raise ValueError(
                f"Role không hợp lệ: {role!r}. "
                f"Phải là một trong {list(self.styles.keys())}"
            )
        style = self.styles[role]
        wav = self.engine.synthesize(text, style, speed=speed)
        return wav

    def synthesize_script(
        self,
        script: List[dict],
        speed_per_role: Optional[Dict[str, float]] = None,
        save_lines_dir: Optional[Path] = None,
        skip_on_error: bool = True,
        empty_cache_every: int = 10,
    ) -> tuple[np.ndarray, List[dict]]:
        """
        Render toàn bộ script -> 1 numpy array audio + stats per line.

        Args:
            script:           list of {id, role, text, pause_after_ms}
            speed_per_role:   override speed per role, vd {"narrator": 0.9}
                              Default: tất cả 1.0
            save_lines_dir:   nếu set, save từng line .wav riêng (debug)
            skip_on_error:    True -> skip line lỗi + tiếp tục.
                              False -> raise lỗi ngay.
            empty_cache_every: gọi torch.cuda.empty_cache() mỗi N lines.

        Returns:
            final_wav: numpy array float32, mono 24kHz
            stats:     list dict per line (id, role, status, speech_dur,
                       pause_dur, time_taken, error?)
        """
        if speed_per_role is None:
            speed_per_role = {}

        if save_lines_dir:
            save_lines_dir = Path(save_lines_dir)
            save_lines_dir.mkdir(parents=True, exist_ok=True)

        audio_chunks: List[np.ndarray] = []
        stats: List[dict] = []

        print(f"\n  Synthesizing {len(script)} lines...")
        pbar = tqdm(script, desc="  Lines", ncols=90)

        for idx, line in enumerate(pbar):
            line_id = line.get("id", idx + 1)
            role = line.get("role", "narrator")
            text = line.get("text", "").strip()
            pause_ms = int(line.get("pause_after_ms", 500))
            speed = float(speed_per_role.get(role, 1.0))

            stat = {
                "id": line_id,
                "role": role,
                "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
                "speed": speed,
                "pause_ms": pause_ms,
                "status": "pending",
                "speech_dur_sec": 0.0,
                "time_taken_sec": 0.0,
                "error": None,
            }

            # ===== Skip empty =====
            if not text:
                stat["status"] = "skipped_empty"
                stats.append(stat)
                pbar.set_postfix_str(f"id={line_id} EMPTY skip")
                # Vẫn chèn silence để giữ đúng thời gian
                n_samples = int(pause_ms / 1000.0 * SR)
                audio_chunks.append(np.zeros(n_samples, dtype=np.float32))
                continue

            # ===== Synthesize =====
            t0 = time.time()
            try:
                wav = self.synthesize_line(text, role, speed=speed)
                stat["time_taken_sec"] = round(time.time() - t0, 3)
                stat["speech_dur_sec"] = round(len(wav) / SR, 3)
                stat["status"] = "ok"

                audio_chunks.append(wav.astype(np.float32))

                if save_lines_dir:
                    line_path = save_lines_dir / f"line_{line_id:04d}_{role}.wav"
                    sf.write(str(line_path), wav, SR)

            except Exception as e:
                err_msg = f"{type(e).__name__}: {e}"
                stat["status"] = "failed"
                stat["error"] = err_msg
                stat["time_taken_sec"] = round(time.time() - t0, 3)
                pbar.set_postfix_str(f"id={line_id} FAIL")
                if skip_on_error:
                    print(f"\n    ⚠️  Line {line_id} fail ({err_msg}). SKIP, continue...")
                    # Chèn silence ngắn 200ms thay cho audio bị mất
                    audio_chunks.append(np.zeros(int(0.2 * SR), dtype=np.float32))
                else:
                    raise

            # ===== Silence padding =====
            n_samples = int(pause_ms / 1000.0 * SR)
            if n_samples > 0:
                audio_chunks.append(np.zeros(n_samples, dtype=np.float32))

            stats.append(stat)
            pbar.set_postfix_str(f"id={line_id} {role[:8]} {stat['time_taken_sec']:.1f}s")

            # ===== Memory hygiene cho 3050Ti 4GB =====
            if (idx + 1) % empty_cache_every == 0:
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        pbar.close()

        # ===== Concat final =====
        print(f"\n  Concatenating {len(audio_chunks)} chunks...")
        final_wav = np.concatenate(audio_chunks).astype(np.float32)
        return final_wav, stats


# ============================================================
# 2. HELPERS
# ============================================================
def load_script(script_path: Path) -> List[dict]:
    """Load script.json -> list of dict. Validate format cơ bản."""
    if not script_path.exists():
        raise FileNotFoundError(f"Script file không tồn tại: {script_path}")

    with open(script_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"Script phải là JSON array, got {type(data).__name__}. "
            "Bạn có nhầm với file .metadata.json không?"
        )

    if not data:
        raise ValueError(f"Script rỗng: {script_path}")

    # Validate fields cần thiết
    required = {"id", "role", "text", "pause_after_ms"}
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise ValueError(f"Item {idx} không phải dict: {item}")
        missing = required - set(item.keys())
        if missing:
            raise ValueError(f"Item {idx} thiếu fields: {missing}")

    return data


def normalize_audio(wav: np.ndarray, target_peak: float = 0.95) -> np.ndarray:
    """Normalize peak amplitude. Tránh clipping."""
    peak = float(np.abs(wav).max())
    if peak < 1e-9:
        return wav
    return wav / peak * target_peak


def print_summary(stats: List[dict], final_wav: np.ndarray) -> dict:
    """In summary + return dict để save vào metadata."""
    total_lines = len(stats)
    ok_lines = sum(1 for s in stats if s["status"] == "ok")
    failed = [s for s in stats if s["status"] == "failed"]
    skipped = [s for s in stats if s["status"] == "skipped_empty"]

    total_speech_sec = sum(s["speech_dur_sec"] for s in stats)
    total_time_sec = sum(s["time_taken_sec"] for s in stats)
    total_audio_sec = len(final_wav) / SR

    role_counts: Dict[str, int] = {}
    for s in stats:
        role_counts[s["role"]] = role_counts.get(s["role"], 0) + 1

    print(f"\n  Lines total       : {total_lines}")
    print(f"    OK              : {ok_lines}")
    print(f"    Failed          : {len(failed)}")
    print(f"    Empty (skipped) : {len(skipped)}")
    print(f"  Roles distribution: {role_counts}")
    print(f"  Total speech      : {total_speech_sec:.2f}s")
    print(f"  Total audio out   : {total_audio_sec:.2f}s ({total_audio_sec/60:.2f} min)")
    print(f"  Total synth time  : {total_time_sec:.2f}s")
    if total_speech_sec > 0:
        rtf = total_time_sec / total_speech_sec
        print(f"  Overall RTF       : {rtf:.3f}x (lower = faster than realtime)")

    if failed:
        print(f"\n  ⚠️  Failed lines:")
        for s in failed[:10]:
            print(f"    [id={s['id']}] {s['error']}")
        if len(failed) > 10:
            print(f"    ... và {len(failed) - 10} dòng nữa")

    return {
        "total_lines": total_lines,
        "ok_lines": ok_lines,
        "failed_lines": len(failed),
        "skipped_empty": len(skipped),
        "role_counts": role_counts,
        "total_speech_sec": round(total_speech_sec, 2),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_audio_min": round(total_audio_sec / 60, 2),
        "total_synth_time_sec": round(total_time_sec, 2),
        "overall_rtf": round(total_time_sec / total_speech_sec, 3) if total_speech_sec > 0 else None,
    }


# ============================================================
# 3. MAIN
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
    # Content
    parser.add_argument("--script", type=str, default=None,
                        help="[tts.script] script.json (output từ D2)")
    parser.add_argument("--output", type=str, default=None,
                        help="[tts.output] file .wav audiobook xuất ra")
    parser.add_argument("--male-ref", type=str, default=None,
                        help="[references.male_ref] giọng Ngạn (narrator + character_male)")
    parser.add_argument("--female-ref", type=str, default=None,
                        help="[references.female_ref] giọng nữ (character_female)")
    # Engine
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="[engine.checkpoint] path tới epoch_*.pth")
    parser.add_argument("--repo", type=str, default=None,
                        help="[engine.repo] folder StyleTTS2-lite")
    parser.add_argument("--model-config", type=str, default=None,
                        help="[engine.model_config] config.yaml KIẾN TRÚC model (khác file --config)")
    parser.add_argument("--device", type=str, default=None,
                        help="[engine.device] auto | cuda | cpu")
    parser.add_argument("--fp16", action=argparse.BooleanOptionalAction, default=None,
                        help="[engine.use_fp16] --fp16 / --no-fp16")
    # Speed
    parser.add_argument("--narrator-speed", type=float, default=None,
                        help="[tts.narrator_speed] speed role narrator (vd 0.9 cho dramatic)")
    parser.add_argument("--character-speed", type=float, default=None,
                        help="[tts.character_speed] speed character_male + character_female")
    # compute_style
    parser.add_argument("--denoise", type=float, default=None,
                        help="[tts.denoise] mức blend noisereduce khi compute_style (0-1)")
    parser.add_argument("--split-dur", type=float, default=None,
                        help="[tts.split_dur] chia ref thành đoạn N giây để avg style")
    # Toggles
    parser.add_argument("--save-lines", action=argparse.BooleanOptionalAction, default=None,
                        help="[tts.save_lines] xuất thêm từng line .wav riêng")
    parser.add_argument("--skip-on-error", action=argparse.BooleanOptionalAction, default=None,
                        help="[tts.skip_on_error] bỏ qua line lỗi & tiếp tục")
    parser.add_argument("--normalize", action=argparse.BooleanOptionalAction, default=None,
                        help="[tts.normalize] normalize peak về 0.95")

    args = parser.parse_args()

    print("=" * 60)
    print("D3 — TTS GENERATOR (Audiobook synthesis)")
    print("=" * 60)

    # ===== Resolve config (CLI > YAML > default) =====
    cfg = load_config(args.config)

    script_path = resolve_path(cfg_value(cfg, "tts", "script", args.script))
    output_path = resolve_path(cfg_value(cfg, "tts", "output", args.output))
    male_ref = reference_path(cfg, args, "male_ref")
    female_ref = reference_path(cfg, args, "female_ref")

    narrator_speed = cfg_value(cfg, "tts", "narrator_speed", args.narrator_speed)
    character_speed = cfg_value(cfg, "tts", "character_speed", args.character_speed)
    denoise = cfg_value(cfg, "tts", "denoise", args.denoise)
    split_dur = cfg_value(cfg, "tts", "split_dur", args.split_dur)
    save_lines = cfg_value(cfg, "tts", "save_lines", args.save_lines)
    skip_on_error = cfg_value(cfg, "tts", "skip_on_error", args.skip_on_error)
    do_normalize = cfg_value(cfg, "tts", "normalize", args.normalize)

    eng_kwargs = engine_kwargs(cfg, args)

    print(f"Script      : {script_path}")
    print(f"Male ref    : {male_ref}")
    print(f"Female ref  : {female_ref}")
    print(f"Output      : {output_path}")
    print(f"Checkpoint  : {eng_kwargs['checkpoint_path']}")
    print(f"Narrator spd: {narrator_speed}")
    print(f"Character sp: {character_speed}")

    # ===== [1/5] Load script =====
    print(f"\n[1/5] Loading script...")
    try:
        script = load_script(script_path)
    except Exception as e:
        print(f"❌ {e}")
        sys.exit(1)
    print(f"  Loaded {len(script)} lines")

    # ===== [2/5] Load engine =====
    print(f"\n[2/5] Loading inference engine...")
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

    # ===== [3/5] Build synthesizer (compute 2 styles) =====
    print(f"\n[3/5] Building AudiobookSynthesizer + computing styles...")
    try:
        synth = AudiobookSynthesizer(
            engine=engine,
            male_ref_path=male_ref,
            female_ref_path=female_ref,
            denoise=denoise,
            split_dur=split_dur,
        )
    except Exception as e:
        print(f"❌ Synthesizer init FAIL: {type(e).__name__}: {e}")
        sys.exit(1)

    # ===== [4/5] Synthesize script =====
    print(f"\n[4/5] Synthesizing audiobook...")
    speed_per_role = {
        "narrator": narrator_speed,
        "character_male": character_speed,
        "character_female": character_speed,
    }
    save_lines_dir = None
    if save_lines:
        save_lines_dir = output_path.parent / f"{output_path.stem}_lines"

    t_start = time.time()
    final_wav, stats = synth.synthesize_script(
        script=script,
        speed_per_role=speed_per_role,
        save_lines_dir=save_lines_dir,
        skip_on_error=skip_on_error,
    )
    total_time = time.time() - t_start

    # Normalize
    if do_normalize:
        final_wav = normalize_audio(final_wav, target_peak=0.95)

    # ===== [5/5] Save + summary =====
    print(f"\n[5/5] Saving outputs...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), final_wav, SR)
    print(f"  Audiobook : {output_path}  ({output_path.stat().st_size/1e6:.1f} MB)")

    summary = print_summary(stats, final_wav)
    summary["total_wall_time_sec"] = round(total_time, 2)

    # Save metadata
    metadata_path = output_path.with_suffix(".metadata.json")
    metadata = {
        "audiobook_file": str(output_path),
        "script_file": str(script_path),
        "male_ref": str(male_ref),
        "female_ref": str(female_ref),
        "checkpoint": str(eng_kwargs["checkpoint_path"]),
        "timestamp": datetime.now().isoformat(),
        "engine_info": engine.info(),
        "speed_per_role": speed_per_role,
        "summary": summary,
        "per_line_stats": stats,
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"  Metadata  : {metadata_path}")

    if save_lines_dir:
        n_line_files = len(list(save_lines_dir.glob("*.wav")))
        print(f"  Per-line  : {save_lines_dir}  ({n_line_files} files)")

    # ===== Final =====
    print(f"\n{'=' * 60}")
    print(f"✅ AUDIOBOOK SYNTHESIS COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Wall time   : {total_time:.1f}s")
    print(f"  Audio dur   : {summary['total_audio_sec']:.1f}s ({summary['total_audio_min']:.1f} min)")
    print(f"  Real-time x : {summary['total_audio_sec']/total_time:.2f}x (audio_dur / synth_time)")
    print(f"\n👉 Mở file: {output_path}")
    print(f"   Nghe + verify chất lượng. Nếu tốt -> sẵn sàng cho UI demo.")

if __name__ == "__main__":
    main()