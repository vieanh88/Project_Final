"""
=============================================================
  STEP 2 (A2): VALIDATE VOCAB + BUILD FINAL FILELIST
=============================================================
Mục tiêu:
  - Đọc 2 file phoneme_raw từ A1 (output cũ).
  - Build symbol_dict ĐÚNG CÁCH như inference.py của lite-vi:
        symbols = pad + punctuation + letters + letters_ipa + extend
        symbol_dict = {ch: idx, ...}
  - Validate từng ký tự trong phoneme — nếu có ký tự lạ KHÔNG trong
    vocab 189, KeyError sẽ bị TextCleaner silently skip lúc training
    -> alignment sai. Ta phải catch ở bước này.
  - Normalize path:
        'output_dataset\\wavs\\ngan_00001.wav'  ->  'wavs/ngan_00001.wav'
    (strip prefix 'output_dataset/' và đổi backslash -> forward slash,
    để filelist khớp với root_path = 'wavs_root/' trên Kaggle/training).
  - Filter audio < 0.5s (BatchSampler skip audio < 20 mel frames ≈ 0.25s,
    ta cut sớm hơn để margin an toàn).
  - Output: ngan_train_lite.txt + ngan_val_lite.txt — filelist FINAL.

Cách chạy (từ root TTS_StyleTTS2-lite-vi/):
    python -X utf8 data_pipeline/prepare_ngan_lite/step2_make_filelist_lite.py

Hoặc kèm flag tắt audio check (chạy nhanh hơn):
    python -X utf8 step2_make_filelist_lite.py --no-audio-check
=============================================================
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Set, Dict, Tuple, List
from tqdm import tqdm
import yaml
import soundfile as sf

# Windows UTF-8 fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"


# ============================================================
# SECTION 1: PATH RESOLUTION
# ----------------------------------------------------------------
# Cấu trúc theo confirm:
#   Project_Final/
#   ├── data/StyleTTS2_preprocess/output_dataset/
#   │   ├── filelist_train.txt
#   │   ├── filelist_val.txt
#   │   └── wavs/                       <-- audio files
#   └── TTS_StyleTTS2-lite-vi/
#       ├── data_pipeline/prepare_ngan_lite/
#       │   └── step2_make_filelist_lite.py    <-- FILE NÀY
#       ├── output/
#       │   ├── ngan_train_phoneme_raw.txt    <-- INPUT (từ A1)
#       │   ├── ngan_val_phoneme_raw.txt
#       │   ├── ngan_train_lite.txt           <-- OUTPUT (FINAL)
#       │   └── ngan_val_lite.txt
#       ├── configs/
#       │   └── config_ngan_kaggle.yml         <-- (chưa tạo, dùng template)
#       └── logs/step2_make_filelist_lite.log
# ============================================================
SCRIPT_PATH = Path(__file__).resolve()
PREPARE_DIR = SCRIPT_PATH.parent
DATA_PIPELINE_DIR = PREPARE_DIR.parent
PROJECT_ROOT = DATA_PIPELINE_DIR.parent          # TTS_StyleTTS2-lite-vi/
PROJECT_FINAL_ROOT = PROJECT_ROOT.parent         # Project_Final/

INPUT_DIR  = PROJECT_ROOT / "output"             # phoneme_raw từ A1
OUTPUT_DIR = PROJECT_ROOT / "output"             # filelist final (cùng folder)
LOG_DIR    = PROJECT_ROOT / "logs"

# Audio root để check duration thực tế.
WAVS_BASE_DIR = (
    PROJECT_FINAL_ROOT / "data" / "StyleTTS2_preprocess" / "output_dataset"
)

INPUT_FILES = {
    "train": INPUT_DIR / "ngan_train_phoneme_raw.txt",
    "val":   INPUT_DIR / "ngan_val_phoneme_raw.txt",
}
OUTPUT_FILES = {
    "train": OUTPUT_DIR / "ngan_train_lite.txt",
    "val":   OUTPUT_DIR / "ngan_val_lite.txt",
}

# Config path: file config.yaml của StyleTTS2-lite-vi (download từ HF).
# User cần đặt config.yaml vào thư mục này hoặc cung cấp đường dẫn.
# Default: tìm ở 2 vị trí phổ biến.
CONFIG_CANDIDATES = [
    PROJECT_ROOT / "configs" / "config.yaml",
    PROJECT_ROOT / "configs" / "config_ngan_kaggle.yml",
    PROJECT_ROOT / "models_pretrained" / "config.yaml",  # nếu user đã tải full model
]


# ============================================================
# SECTION 2: LOGGING
# ============================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "step2_make_filelist_lite.log"

    root_logger = logging.getLogger()
    for h in list(root_logger.handlers):
        root_logger.removeHandler(h)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("step2_make_filelist_lite")


# ============================================================
# SECTION 3: BUILD VOCAB FROM CONFIG
# ----------------------------------------------------------------
# Logic này LẤY NGUYÊN từ inference.py của StyleTTS2-lite-vi
# (https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/blob/main/inference.py)
# để đảm bảo vocab build ra HOÀN TOÀN giống với lúc training/inference.
# ============================================================
def build_symbol_dict_from_config(config_path: Path) -> Tuple[Dict[str, int], int]:
    """
    Build symbol_dict GIỐNG HỆT cách inference.py của lite-vi build:

        symbols = (
            list(config['symbol']['pad']) +
            list(config['symbol']['punctuation']) +
            list(config['symbol']['letters']) +
            list(config['symbol']['letters_ipa']) +
            list(config['symbol']['extend'])
        )

    Returns:
        symbol_dict: {char: index}
        n_token: len(symbol_dict) + 1  (cộng 1 theo logic inference.py)
    """
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    sym = config.get("symbol")
    if sym is None:
        raise KeyError(
            f"Config file {config_path} không có key 'symbol'. "
            "Hãy chắc chắn bạn đang dùng config.yaml của StyleTTS2-lite-vi "
            "(download từ HuggingFace dangtr0408/StyleTTS2-lite-vi)."
        )

    required_keys = ["pad", "punctuation", "letters", "letters_ipa", "extend"]
    for k in required_keys:
        if k not in sym:
            raise KeyError(f"Config 'symbol' thiếu key '{k}'")

    symbols = (
        list(sym["pad"])
        + list(sym["punctuation"])
        + list(sym["letters"])
        + list(sym["letters_ipa"])
        + list(sym["extend"])
    )

    symbol_dict = {symbols[i]: i for i in range(len(symbols))}
    n_token = len(symbol_dict) + 1
    return symbol_dict, n_token


def find_config_file(logger: logging.Logger) -> Path:
    """Tìm config.yaml ở các vị trí ứng viên."""
    for cand in CONFIG_CANDIDATES:
        if cand.exists():
            logger.info(f"Đã tìm thấy config: {cand}")
            return cand

    msg = (
        "KHÔNG TÌM THẤY config.yaml của StyleTTS2-lite-vi.\n"
        "Hãy download config.yaml từ:\n"
        "  https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/blob/main/Models/config.yaml\n"
        "Rồi đặt vào MỘT TRONG các đường dẫn:\n"
        + "\n".join(f"  - {p}" for p in CONFIG_CANDIDATES)
    )
    raise FileNotFoundError(msg)


# ============================================================
# SECTION 4: PATH NORMALIZATION
# ============================================================
def normalize_wav_path(raw_path: str) -> str:
    """
    Chuẩn hóa wav_path từ pipeline cũ -> format dùng được trên Kaggle/Linux.

    Ví dụ:
        'output_dataset\\wavs\\ngan_00001.wav'  ->  'wavs/ngan_00001.wav'
        'output_dataset/wavs/ngan_00001.wav'    ->  'wavs/ngan_00001.wav'
        'wavs\\ngan_00001.wav'                  ->  'wavs/ngan_00001.wav'

    Logic:
      1. Đổi backslash '\\' thành forward slash '/'.
      2. Strip prefix 'output_dataset/' nếu có (vì root_path lúc training
         sẽ trỏ trực tiếp vào folder chứa wavs/).
      3. Strip leading slash để giữ định dạng tương đối.
    """
    p = raw_path.replace("\\", "/").strip()

    # Strip prefix output_dataset/ (case-insensitive)
    PREFIX = "output_dataset/"
    if p.lower().startswith(PREFIX):
        p = p[len(PREFIX):]

    p = p.lstrip("/")
    return p


# ============================================================
# SECTION 5: VOCAB VALIDATION
# ============================================================
def validate_phoneme(
    phoneme: str,
    symbol_dict: Dict[str, int],
) -> Tuple[bool, Set[str]]:
    """
    Check tất cả ký tự trong phoneme có nằm trong vocab không.

    Returns:
        (is_valid, unknown_chars_set)
    """
    unknown = set()
    for ch in phoneme:
        if ch not in symbol_dict:
            unknown.add(ch)
    return (len(unknown) == 0), unknown

def normalize_phoneme_for_lite_vocab(phoneme: str) -> str:
    """
    Normalize phoneme sinh bởi espeak-ng để khớp vocab 189 của StyleTTS2-lite-vi.

    Lý do:
      - U+032A '̪' là combining dental bridge below, espeak-ng có thể sinh ra
        trong các cụm như t̪/d̪/n̪, nhưng vocab lite-vi không có ký tự này.
        Xóa nó sẽ biến t̪ -> t, d̪ -> d, n̪ -> n.
      - '-' không nằm trong vocab. Đổi sang space để không nối dính 2 cụm phoneme.
    """
    if not phoneme:
        return ""

    replacements = {
        "\u032A": "",   # '̪' combining dental bridge below
        "-": " ",       # ASCII hyphen
        "\u2010": " ",  # hyphen
        "\u2011": " ",  # non-breaking hyphen
        "\u2012": " ",  # figure dash
        "\u2013": " ",  # en dash
        "\u2014": " ",  # em dash
        "\u2212": " ",  # minus sign
    }

    for src, dst in replacements.items():
        phoneme = phoneme.replace(src, dst)

    phoneme = " ".join(phoneme.split())
    return phoneme

# ============================================================
# SECTION 6: AUDIO DURATION CHECK
# ============================================================
def check_audio_duration(
    raw_wav_path: str,
    wavs_base_dir: Path,
    min_duration_sec: float = 0.5,
) -> Tuple[bool, float]:
    """
    Đọc audio thực tế để check duration.

    raw_wav_path là đường dẫn từ filelist gốc, ví dụ:
        'output_dataset\\wavs\\ngan_00001.wav'

    wavs_base_dir là folder gốc, ví dụ:
        '.../data/StyleTTS2_preprocess/output_dataset/'

    Logic ghép path: lấy raw_wav_path, normalize backslash, ghép với
    parent của wavs_base_dir nếu raw_wav_path đã chứa 'output_dataset/'
    (vì wavs_base_dir đã CÓ 'output_dataset/').

    Returns:
        (is_long_enough, duration_in_seconds)
        Nếu không đọc được file -> (False, 0.0)
    """
    p = raw_wav_path.replace("\\", "/").strip().lstrip("/")

    # Try multiple resolution strategies vì path từ filelist gốc có thể
    # bao gồm hoặc không bao gồm prefix 'output_dataset/'.
    candidates = [
        wavs_base_dir.parent / p,    # nếu p = 'output_dataset/wavs/x.wav'
        wavs_base_dir / p,           # nếu p = 'wavs/x.wav'
    ]
    # Bổ sung: strip prefix 'output_dataset/' và ghép với wavs_base_dir
    PREFIX = "output_dataset/"
    if p.lower().startswith(PREFIX):
        candidates.append(wavs_base_dir / p[len(PREFIX):])

    full_path = None
    for c in candidates:
        if c.exists():
            full_path = c
            break

    if full_path is None:
        return False, 0.0

    try:
        info = sf.info(str(full_path))
        duration = info.frames / info.samplerate
        return (duration >= min_duration_sec), duration
    except Exception:
        return False, 0.0


# ============================================================
# SECTION 7: PROCESS ONE FILELIST
# ============================================================
def process_filelist(
    input_path: Path,
    output_path: Path,
    symbol_dict: Dict[str, int],
    do_audio_check: bool,
    logger: logging.Logger,
) -> Dict[str, int]:
    """
    Đọc file phoneme_raw, validate, normalize, filter, ghi file final.
    """
    if not input_path.exists():
        logger.error(f"Không tìm thấy file: {input_path}")
        return {"ok": 0, "vocab_fail": 0, "audio_short": 0, "audio_missing": 0,
                "format_bad": 0}

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đã load {len(lines)} dòng từ {input_path.name}")

    valid_records = []
    stats = {"ok": 0, "vocab_fail": 0, "audio_short": 0,
             "audio_missing": 0, "format_bad": 0}
    all_unknown_chars: Dict[str, int] = {}  # char -> count
    sample_logged = False

    for line_num, line in enumerate(
        tqdm(lines, desc=f"Validate {input_path.name}", ncols=90), start=1
    ):
        line = line.strip()
        if not line:
            continue

        parts = line.split("|")
        if len(parts) != 2:
            stats["format_bad"] += 1
            continue

        raw_wav_path, phoneme = parts[0].strip(), parts[1].strip()
        if not raw_wav_path or not phoneme:
            stats["format_bad"] += 1
            continue

        # Normalize phoneme để khớp vocab 189 của StyleTTS2-lite-vi
        phoneme = normalize_phoneme_for_lite_vocab(phoneme)
        if not phoneme:
            stats["format_bad"] += 1
            continue

        # 1) Validate vocab character-level
        is_valid, unknown = validate_phoneme(phoneme, symbol_dict)
        if not is_valid:
            stats["vocab_fail"] += 1
            for ch in unknown:
                all_unknown_chars[ch] = all_unknown_chars.get(ch, 0) + 1
            continue

        # 2) Audio duration check (optional)
        if do_audio_check:
            ok_dur, duration = check_audio_duration(
                raw_wav_path, WAVS_BASE_DIR, min_duration_sec=0.5
            )
            if duration == 0.0:
                stats["audio_missing"] += 1
                if stats["audio_missing"] <= 3:
                    logger.warning(
                        f"  Dòng {line_num}: không đọc được audio "
                        f"'{raw_wav_path}' (thử các path candidates đều fail)"
                    )
                continue
            if not ok_dur:
                stats["audio_short"] += 1
                continue

        # 3) Normalize path -> format final cho training
        norm_path = normalize_wav_path(raw_wav_path)

        valid_records.append(f"{norm_path}|{phoneme}")
        stats["ok"] += 1

        if not sample_logged:
            logger.info(
                f"  [SAMPLE] Raw path : {raw_wav_path}\n"
                f"           Norm path: {norm_path}\n"
                f"           Phoneme  : {phoneme[:80]}..."
            )
            sample_logged = True

    # Ghi output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(valid_records))
        if valid_records:
            f.write("\n")

    # Log thống kê
    logger.info(f"  Hoàn tất: {input_path.name}")
    logger.info(f"    OK             : {stats['ok']:>6} dòng")
    logger.info(f"    Format sai     : {stats['format_bad']:>6} dòng")
    logger.info(f"    Vocab fail     : {stats['vocab_fail']:>6} dòng")
    logger.info(f"    Audio missing  : {stats['audio_missing']:>6} dòng")
    logger.info(f"    Audio < 0.5s   : {stats['audio_short']:>6} dòng")
    logger.info(f"    -> Đã ghi: {output_path}")

    if all_unknown_chars:
        # Sort theo số lần xuất hiện (giảm dần)
        sorted_unknown = sorted(
            all_unknown_chars.items(), key=lambda x: -x[1]
        )
        logger.warning(
            f"  Phát hiện {len(all_unknown_chars)} ký tự LẠ "
            f"(không có trong vocab 189):"
        )
        for ch, cnt in sorted_unknown[:20]:
            logger.warning(
                f"    '{ch}' (U+{ord(ch):04X}) — xuất hiện trong "
                f"{cnt} dòng"
            )
        if len(sorted_unknown) > 20:
            logger.warning(f"    ... và {len(sorted_unknown) - 20} ký tự khác.")
        logger.warning(
            "  Nếu bạn thấy nhiều dòng bị skip, có thể text gốc có "
            "ký tự lạ chưa clean hết. Hãy check log và quyết định "
            "có cần re-clean không."
        )

    return stats


# ============================================================
# SECTION 8: MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-audio-check",
        action="store_true",
        help="Tắt việc đọc file audio để check duration (chạy nhanh hơn).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Đường dẫn tới config.yaml của lite-vi (override auto-detect).",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("  STEP 2: BUILD FINAL FILELIST FOR StyleTTS2-lite-vi")
    logger.info("=" * 60)
    logger.info(f"Input dir       : {INPUT_DIR}")
    logger.info(f"Output dir      : {OUTPUT_DIR}")
    logger.info(f"Wavs base dir   : {WAVS_BASE_DIR}")
    logger.info(f"Audio check     : {'OFF' if args.no_audio_check else 'ON'}")
    logger.info("-" * 60)

    # 1) Locate config & build vocab
    if args.config:
        config_path = Path(args.config).resolve()
        if not config_path.exists():
            logger.error(f"Config được chỉ định không tồn tại: {config_path}")
            sys.exit(1)
    else:
        try:
            config_path = find_config_file(logger)
        except FileNotFoundError as e:
            logger.error(str(e))
            sys.exit(1)

    try:
        symbol_dict, n_token = build_symbol_dict_from_config(config_path)
    except Exception as e:
        logger.error(f"Build vocab thất bại: {e}")
        sys.exit(1)

    logger.info(f"Built symbol_dict: {len(symbol_dict)} unique symbols")
    logger.info(f"  n_token (theo logic inference.py = len + 1) = {n_token}")

    # Cảnh báo nếu n_token != 189 — có thể user dùng config sai version
    if n_token != 189:
        logger.warning(
            f"  ! n_token = {n_token}, KHÁC 189 (số chuẩn của lite-vi).\n"
            f"    Nếu bạn dùng config gốc của lite-vi thì điều này là bất thường.\n"
            f"    Hãy kiểm tra lại config file: {config_path}"
        )
    logger.info("-" * 60)

    # 2) Sanity check WAVS_BASE_DIR (chỉ nếu bật audio check)
    if not args.no_audio_check:
        if not WAVS_BASE_DIR.exists():
            logger.error(
                f"WAVS_BASE_DIR không tồn tại: {WAVS_BASE_DIR}\n"
                f"  Bạn có 2 lựa chọn:\n"
                f"  (a) Sửa cấu trúc folder cho đúng (copy data vào "
                f"data/StyleTTS2_preprocess/output_dataset/)\n"
                f"  (b) Chạy lại với flag --no-audio-check để bỏ qua "
                f"việc check duration"
            )
            sys.exit(1)

    # 3) Process từng split
    total = {"ok": 0, "vocab_fail": 0, "audio_short": 0,
             "audio_missing": 0, "format_bad": 0}
    for split_name, in_path in INPUT_FILES.items():
        logger.info(f"\n>>> Xử lý split: {split_name.upper()}")
        out_path = OUTPUT_FILES[split_name]
        stats = process_filelist(
            in_path, out_path, symbol_dict,
            do_audio_check=(not args.no_audio_check),
            logger=logger,
        )
        for k in total:
            total[k] += stats.get(k, 0)

    # 4) Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT")
    logger.info("=" * 60)
    logger.info(f"  Tổng OK        : {total['ok']}")
    logger.info(f"  Format sai     : {total['format_bad']}")
    logger.info(f"  Vocab fail     : {total['vocab_fail']}")
    logger.info(f"  Audio missing  : {total['audio_missing']}")
    logger.info(f"  Audio < 0.5s   : {total['audio_short']}")
    logger.info("")
    logger.info("Bước tiếp theo:")
    logger.info("  - Nếu OK count đủ lớn (~95% input) -> chạy file A3 "
                "(zero-shot test)")
    logger.info("  - Nếu vocab_fail nhiều -> kiểm tra log ký tự lạ phía trên")

if __name__ == "__main__":
    main()