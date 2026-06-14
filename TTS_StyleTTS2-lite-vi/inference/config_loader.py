"""
=============================================================
  CONFIG LOADER — Module dùng chung cho toàn bộ folder inference/
=============================================================
Mục đích:
  Gom TẤT CẢ đường dẫn + tham số của 4 file inference vào 1 file
  YAML duy nhất (inference_config.yaml) để dễ chỉnh sửa.

  Các file D0/D1/D2/D3 import module này để:
    - Load config YAML  (load_config)
    - Resolve đường dẫn (resolve_path)  — tự nối relative path vào PROJECT_ROOT
    - Chọn giá trị theo thứ tự ưu tiên: CLI > config > default (cfg_value)
    - Lấy nhanh tham số engine + reference (engine_kwargs / reference_path)

Thứ tự ưu tiên cho MỌI tham số:
    1. Giá trị truyền qua CLI (vd --checkpoint ...)   → cao nhất
    2. Giá trị trong file YAML (--config ...)
    3. Giá trị mặc định trong DEFAULTS bên dưới       → thấp nhất

Quy ước đường dẫn trong YAML:
    - Có thể dùng / hoặc \\ , đường dẫn TUYỆT ĐỐI (D:\\...) hoặc TƯƠNG ĐỐI.
    - Đường dẫn tương đối được tính từ PROJECT_ROOT = thư mục
      TTS_StyleTTS2-lite-vi/  (KHÔNG phải thư mục đang chạy lệnh).
=============================================================
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional


# ============================================================
# Project layout
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent          # .../inference/
PROJECT_ROOT = SCRIPT_DIR.parent                       # .../TTS_StyleTTS2-lite-vi/

# File config mặc định nếu người dùng không truyền --config
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "inference_config.yaml"


# ============================================================
# DEFAULTS — fallback cuối cùng khi YAML thiếu key / không có --config
# (Khớp với layout thực tế: checkpoint + reference nằm trong StyleTTS2-lite/Models)
# ============================================================
DEFAULTS: dict[str, dict[str, Any]] = {
    # Dùng chung bởi D0 (engine), D1 (verify), D3 (tts)
    "engine": {
        "checkpoint":   "StyleTTS2-lite/Models/Finetune/epoch_00025.pth",
        "repo":         "StyleTTS2-lite",
        "model_config": "configs/config.yaml",
        "device":       "auto",     # auto | cuda | cpu
        "use_fp16":     True,
    },
    # 2 file audio reference (giọng mẫu)
    "references": {
        "male_ref":   "StyleTTS2-lite/Models/references/male_reference.wav",
        # File female mặc định nằm NGOÀI project (ở TTS_StyleTTS2). Đổi nếu cần.
        "female_ref": "StyleTTS2-lite/Models/references/female_reference.wav",
    },
    # D2 — nlp_generator.py
    "nlp": {
        "input":       "data/raw_stories/horror_story_test.txt",
        "output":      None,                # None -> tự sinh output/nlp/<stem>.json
        "model":       "gemini-3.1-flash",
        "env":         "../.env",           # = Project_Final/.env
        "chunk_size":  8000,
        "max_retries": 3,
        "thinking":    True,                # False = tắt thinking (nhanh hơn ~2x)
        "dry_run":     False,
    },
    # D3 — tts_generator.py
    "tts": {
        "script":          "output/nlp/horror_story_test.json",
        "output":          "output/audiobooks/horror_story_test.wav",
        "narrator_speed":  1.0,
        "character_speed": 1.0,
        "denoise":         0.3,             # compute_style: % blend noisereduce (0-1)
        "split_dur":       2.0,             # compute_style: chia ref thành đoạn N giây
        "save_lines":      False,
        "skip_on_error":   True,
        "normalize":       True,
    },
    # D1 — download_female_ref.py (verify reference)
    "verify_ref": {
        "skip_synthesize": False,
    },
    # D0 — inference_engine.py (smoke test khi chạy trực tiếp)
    "smoke": {
        "text":   "Đêm hôm ấy, trời tối đen như mực, không một tiếng động.",
        "output": "output/smoke_test.wav",
    },
}


# ============================================================
# Load / resolve helpers
# ============================================================
def load_config(config_path: Optional[str | Path] = None) -> dict:
    """
    Load file YAML config -> dict. Robust:
      - config_path = None  -> dùng DEFAULT_CONFIG_PATH
      - file không tồn tại  -> trả về {} (sẽ rơi về DEFAULTS)
      - file rỗng           -> trả về {}
    """
    import yaml

    path = Path(config_path).resolve() if config_path else DEFAULT_CONFIG_PATH
    if not path.exists():
        print(f"  ⚠️  Config không tìm thấy ở {path} — dùng giá trị DEFAULTS.")
        return {}

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Config YAML phải là object (key: value), got {type(data).__name__}: {path}"
        )
    print(f"  Loaded config: {path}")
    return data


def resolve_path(value: Optional[str | Path], base: Path = PROJECT_ROOT) -> Optional[Path]:
    """
    Đổi string đường dẫn -> Path tuyệt đối.
      - None        -> None  (giữ nguyên để caller xử lý, vd nlp.output=None)
      - tuyệt đối   -> giữ nguyên
      - tương đối   -> nối vào base (mặc định PROJECT_ROOT)
    """
    if value is None:
        return None
    p = Path(value)
    if not p.is_absolute():
        p = base / p
    return p.resolve()


def pick(*values: Any) -> Any:
    """Trả về giá trị non-None ĐẦU TIÊN. Dùng cho ưu tiên CLI > config > default."""
    for v in values:
        if v is not None:
            return v
    return None


def cfg_value(cfg: dict, section: str, key: str, cli: Any = None) -> Any:
    """
    Resolve 1 tham số theo ưu tiên: CLI > cfg[section][key] > DEFAULTS[section][key].

    Args:
        cfg:     dict trả về từ load_config()
        section: vd "engine", "tts", "nlp", "references", ...
        key:     tên tham số trong section
        cli:     giá trị từ argparse (None nếu user không truyền)
    """
    config_val = cfg.get(section, {}).get(key) if isinstance(cfg.get(section), dict) else None
    default_val = DEFAULTS.get(section, {}).get(key)
    return pick(cli, config_val, default_val)


# ============================================================
# High-level resolvers (giữ main() của các script gọn gàng)
# ============================================================
def engine_kwargs(cfg: dict, args) -> dict:
    """
    Build kwargs để khởi tạo StyleTTS2LiteVNInference.
    Đọc CLI args (args.checkpoint/repo/model_config/device/fp16) nếu có,
    fallback về config rồi DEFAULTS. Đường dẫn đã resolve sang absolute Path.
    """
    return {
        "checkpoint_path": resolve_path(cfg_value(cfg, "engine", "checkpoint",   getattr(args, "checkpoint", None))),
        "repo_root":       resolve_path(cfg_value(cfg, "engine", "repo",         getattr(args, "repo", None))),
        "config_path":     resolve_path(cfg_value(cfg, "engine", "model_config", getattr(args, "model_config", None))),
        "device":          cfg_value(cfg, "engine", "device",   getattr(args, "device", None)),
        "use_fp16":        cfg_value(cfg, "engine", "use_fp16", getattr(args, "fp16", None)),
    }


def reference_path(cfg: dict, args, which: str) -> Optional[Path]:
    """Lấy đường dẫn reference đã resolve. which ∈ {'male_ref', 'female_ref'}."""
    cli = getattr(args, which, None)
    return resolve_path(cfg_value(cfg, "references", which, cli))
