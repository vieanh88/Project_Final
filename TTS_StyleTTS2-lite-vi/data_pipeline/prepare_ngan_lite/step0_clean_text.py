"""
=============================================================
  STEP 0 (A0): CLEAN TEXT GỐC TRƯỚC KHI PHONEMIZE
=============================================================
Mục tiêu:
  Clean filelist text gốc tiếng Việt của Ngạn để loại bỏ:
    1. Chữ số (1900, 5.000, 23000 giây, ...)  -> tier "digit"
    2. Từ tiếng Anh trong outro YouTube (subscribe/video/school)
       -> tier 1: xóa cả dòng
    3. Từ vay mượn tiếng Anh -> tier 2: map sang phiên âm tiếng Việt
       theo file english_mapping.txt
    4. Từ tiếng Anh khác chưa map -> tier 3: xóa khỏi câu
    5. Brackets, dấu đặc biệt khác

Input :  data/StyleTTS2_preprocess/output_dataset/filelist_{train,val}.txt
         (text gốc tiếng Việt có dấu, format: wav_path|text)
Output:  TTS_StyleTTS2-lite-vi/output/filelist_{train,val}_clean.txt
         (text đã clean, cùng format)

Sau bước này, A1 (step1_rephonemize_lite.py) cần SỬA NHẸ để đọc input
từ output/filelist_{train,val}_clean.txt thay vì filelist_{train,val}.txt.

Chạy:
    python -X utf8 step0_clean_text.py                 # dùng config mặc định
    python -X utf8 step0_clean_text.py --config <path> # custom config

Workflow đề xuất:
    1. Chỉnh config -> mode: "preview" -> chạy -> check terminal
    2. Nếu OK -> đổi mode: "apply" -> chạy lại -> file output được ghi
=============================================================
"""

import os
import sys
import re
import logging
import argparse
import unicodedata
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
import yaml
from tqdm import tqdm

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
# SECTION 1: PATHS
# ============================================================
SCRIPT_PATH = Path(__file__).resolve()
PREPARE_DIR = SCRIPT_PATH.parent                    # data_pipeline/prepare_ngan_lite/
DATA_PIPELINE_DIR = PREPARE_DIR.parent
PROJECT_ROOT = DATA_PIPELINE_DIR.parent              # TTS_StyleTTS2-lite-vi/
LOG_DIR = PROJECT_ROOT / "logs"


# ============================================================
# SECTION 2: VIETNAMESE DIACRITIC DETECTION
# ----------------------------------------------------------------
# Để phát hiện 1 từ "có vẻ là tiếng Anh", ta check xem nó có chứa
# ký tự có dấu tiếng Việt không. Cách robust nhất: dùng Unicode
# normalization NFD để tách combining marks ra, rồi check.
#
# VD: 'à' (U+00E0) -> NFD -> 'a' + '\u0300' (combining grave accent)
#     'a' không có combining mark -> "no diacritic"
# ============================================================
def has_vietnamese_diacritic(word: str) -> bool:
    """
    Trả về True nếu từ chứa ÍT NHẤT 1 ký tự có dấu tiếng Việt.

    Cách check:
      - Normalize word sang NFD (decompose).
      - Nếu có ANY ký tự thuộc category 'Mn' (Mark, Nonspacing) -> có dấu.
      - Hoặc nếu có 'đ' / 'Đ' (không phân tách trong NFD) -> có dấu.
    """
    if "đ" in word.lower():
        return True
    decomposed = unicodedata.normalize("NFD", word)
    for ch in decomposed:
        if unicodedata.category(ch) == "Mn":
            return True
    return False


def is_likely_english(token: str, min_length: int, mapping_keys_lower: Set[str]) -> bool:
    """
    Một token "có vẻ là tiếng Anh" nếu:
      1. Độ dài >= min_length
      2. Chỉ chứa chữ cái ASCII (a-z, A-Z)
      3. KHÔNG chứa dấu tiếng Việt
      4. KHÔNG nằm trong mapping (vì những từ này đã được handle ở Tier 2)
    """
    if len(token) < min_length:
        return False
    if not token.isascii():
        return False
    if not token.isalpha():
        return False
    if has_vietnamese_diacritic(token):
        return False
    if token.lower() in mapping_keys_lower:
        return False
    return True


# ============================================================
# SECTION 3: PARSE ENGLISH MAPPING FILE
# ============================================================
def parse_english_mapping(file_path: Path) -> Tuple[List[str], Dict[str, str]]:
    """
    Parse file english_mapping.txt với format:
      - Dòng KHÔNG có dấu phẩy = delete keyword (Tier 1)
      - Dòng CÓ dấu phẩy       = mapping English, Vietnamese (Tier 2)

    Returns:
        delete_keywords: list[str]  (lowercase)
        mapping:         dict[str_lowercase -> vietnamese_str]
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy mapping file: {file_path}\n"
            f"Hãy đặt file english_mapping.txt vào thư mục:\n"
            f"  {file_path.parent}"
        )

    delete_keywords: List[str] = []
    mapping: Dict[str, str] = {}

    with open(file_path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "," in line:
                eng, vi = line.split(",", 1)
                eng = eng.strip().lower()
                vi = vi.strip()
                if eng and vi:
                    mapping[eng] = vi
            else:
                delete_keywords.append(line.lower())

    return delete_keywords, mapping


# ============================================================
# SECTION 4: PROCESS ONE TEXT LINE
# ============================================================
class CleanStats:
    """Đếm số dòng được xử lý theo từng lý do."""
    def __init__(self):
        self.total = 0
        self.kept = 0
        self.deleted_tier1 = 0
        self.deleted_tier3 = 0
        self.deleted_empty = 0
        self.modified = 0
        # Đếm các từ tiếng Anh "lạ" đã bị remove
        self.unknown_english_words: Dict[str, int] = {}


def process_text(
    text: str,
    delete_keywords: List[str],
    mapping: Dict[str, str],
    sorted_mapping_keys: List[str],
    mapping_keys_lower: Set[str],
    cfg: dict,
    stats: CleanStats,
) -> Tuple[Optional[str], List[str]]:
    """
    Xử lý 1 text string.

    Pipeline:
      1. Tier 1: nếu có delete keyword (word-boundary) -> return None.
      2. Tier 2: replace English -> Vietnamese (phrase dài match trước).
      3. Remove digits (nếu config bật).
      4. Remove punctuation đặc biệt (brackets, *, &, ...).
      5. Tier 3: detect & handle các token tiếng Anh "lạ" còn lại.
      6. Normalize whitespace.

    Returns:
        (cleaned_text or None nếu bị xóa, list of debug notes)
    """
    notes = []
    original = text

    # ---------- Tier 1: Delete keywords ----------
    text_lower_check = text.lower()
    for kw in delete_keywords:
        # \b không work tốt với Unicode -> dùng manual word boundary
        # bằng cách check ký tự xung quanh không phải letter
        pattern = r"(?<![a-zA-Z])" + re.escape(kw) + r"(?![a-zA-Z])"
        if re.search(pattern, text_lower_check):
            notes.append(f"DELETE_TIER1: keyword='{kw}'")
            return None, notes

    # ---------- Tier 2: Replace English -> Vietnamese ----------
    # Quan trọng: sort by length DESC để match "shopping center" trước "shopping"
    flags = re.IGNORECASE if cfg.get("case_insensitive_mapping", True) else 0
    for eng in sorted_mapping_keys:
        pattern = re.compile(
            r"(?<![a-zA-Z])" + re.escape(eng) + r"(?![a-zA-Z])",
            flags,
        )
        if pattern.search(text):
            text = pattern.sub(mapping[eng], text)

    # ---------- Remove digits ----------
    if cfg.get("remove_digits", True):
        text = re.sub(r"\d+", " ", text)

    # ---------- Remove punctuation đặc biệt ----------
    punct_to_remove = cfg.get("punctuation_to_remove", "[]()*&%-")
    for p in punct_to_remove:
        text = text.replace(p, " ")

    # ---------- Tier 3: Handle unknown English words ----------
    unknown_action = cfg.get("unknown_english_action", "remove_word")
    min_len = cfg.get("unknown_english_min_length", 4)

    if unknown_action != "keep_warn":
        # Tách thành các token (giữ punctuation câu)
        # Cách tách: split by whitespace, sau đó với mỗi token strip
        # punctuation đầu/cuối để check.
        tokens = text.split()
        new_tokens = []
        for tok in tokens:
            # Strip punctuation đầu/cuối để check token core
            core = re.sub(r"^[^\w]+|[^\w]+$", "", tok, flags=re.UNICODE)
            if is_likely_english(core, min_len, mapping_keys_lower):
                if unknown_action == "delete_line":
                    notes.append(f"DELETE_TIER3: unknown='{core}'")
                    return None, notes
                elif unknown_action == "remove_word":
                    notes.append(f"REMOVE_TIER3: '{core}'")
                    stats.unknown_english_words[core] = (
                        stats.unknown_english_words.get(core, 0) + 1
                    )
                    continue  # skip token này
            new_tokens.append(tok)
        text = " ".join(new_tokens)

    # ---------- Normalize whitespace & punctuation spacing ----------
    text = re.sub(r"\s+", " ", text).strip()
    # Gộp space trước dấu câu: "xin chào ." -> "xin chào."
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    # Đảm bảo có space sau dấu câu (nếu thiếu): "xin,chào" -> "xin, chào"
    text = re.sub(r"([,.!?;:])([^\s])", r"\1 \2", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text:
        notes.append("DELETE_EMPTY: cleaning made text empty")
        return None, notes

    if text != original.strip():
        notes.append("MODIFIED")

    return text, notes


# ============================================================
# SECTION 5: PROCESS ONE FILE
# ============================================================
def process_file(
    in_path: Path,
    out_path: Path,
    delete_keywords: List[str],
    mapping: Dict[str, str],
    cfg: dict,
    logger: logging.Logger,
    write_output: bool,
) -> CleanStats:
    """
    Xử lý 1 file filelist.

    Nếu write_output = False (preview mode): chỉ in ra terminal, không ghi file.
    Nếu write_output = True: ghi file output đầy đủ.
    """
    stats = CleanStats()

    if not in_path.exists():
        logger.error(f"Không tìm thấy file: {in_path}")
        return stats

    sorted_mapping_keys = sorted(mapping.keys(), key=len, reverse=True)
    mapping_keys_lower = set(mapping.keys())

    with open(in_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đã load {len(lines)} dòng từ {in_path.name}")

    output_lines = []
    preview_lines_limit = cfg.get("preview_lines", 50)
    preview_only_changed = cfg.get("preview_show_only_changed", True)
    preview_printed = 0

    desc = "Apply " if write_output else "Preview "
    for line_num, raw_line in enumerate(
        tqdm(lines, desc=f"{desc}{in_path.name}", ncols=90), start=1
    ):
        line = raw_line.strip()
        stats.total += 1

        if not line:
            continue

        parts = line.split("|", 1)
        if len(parts) != 2:
            # Format sai -> giữ nguyên để debug, log
            if not write_output:
                logger.warning(f"  Dòng {line_num}: format sai (cần 2 cột): {line[:80]}")
            continue

        wav_path, text = parts[0].strip(), parts[1].strip()
        if not wav_path or not text:
            continue

        cleaned, notes = process_text(
            text, delete_keywords, mapping,
            sorted_mapping_keys, mapping_keys_lower,
            cfg, stats,
        )

        # Đếm theo lý do
        if cleaned is None:
            if any(n.startswith("DELETE_TIER1") for n in notes):
                stats.deleted_tier1 += 1
            elif any(n.startswith("DELETE_TIER3") for n in notes):
                stats.deleted_tier3 += 1
            elif any(n.startswith("DELETE_EMPTY") for n in notes):
                stats.deleted_empty += 1
        else:
            stats.kept += 1
            if any(n == "MODIFIED" for n in notes):
                stats.modified += 1
            output_lines.append(f"{wav_path}|{cleaned}")

        # Preview output
        if not write_output and preview_printed < preview_lines_limit:
            is_changed = (cleaned is None) or (
                cleaned is not None and any(
                    n == "MODIFIED" or n.startswith("REMOVE_TIER3") for n in notes
                )
            )
            if (not preview_only_changed) or is_changed:
                preview_printed += 1
                status = (
                    "[DELETED]" if cleaned is None
                    else "[MODIFIED]" if any(n == "MODIFIED" for n in notes)
                    else "[UNCHANGED]"
                )
                print(f"\n--- Line {line_num} {status} ---")
                print(f"  ORIG: {text[:200]}")
                if cleaned is not None:
                    print(f"  NEW : {cleaned[:200]}")
                # In notes liên quan
                interesting_notes = [
                    n for n in notes
                    if not n == "MODIFIED"
                ]
                if interesting_notes:
                    print(f"  NOTE: {'; '.join(interesting_notes)}")

    # Ghi file (chỉ apply mode)
    if write_output:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("\n".join(output_lines))
            if output_lines:
                f.write("\n")
        logger.info(f"  -> Đã ghi: {out_path}")
    else:
        logger.info(
            f"  [PREVIEW MODE] Đã in {preview_printed} dòng đầu (giới hạn "
            f"{preview_lines_limit})"
        )

    return stats


# ============================================================
# SECTION 6: LOGGING
# ============================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "step0_clean_text.log"

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
    return logging.getLogger("step0_clean_text")


# ============================================================
# SECTION 7: MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=str,
        default=str(PREPARE_DIR / "step0_clean_text.config.yaml"),
        help="Đường dẫn tới config yaml.",
    )
    parser.add_argument(
        "--force-apply",
        action="store_true",
        help="Override mode -> 'apply' bất kể config (tiện CLI).",
    )
    parser.add_argument(
        "--force-preview",
        action="store_true",
        help="Override mode -> 'preview' bất kể config.",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("  STEP 0: CLEAN TEXT (pre-phonemize)")
    logger.info("=" * 60)

    # Load config
    config_path = Path(args.config).resolve()
    if not config_path.exists():
        logger.error(f"Config không tồn tại: {config_path}")
        sys.exit(1)
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    # Override mode nếu có flag
    if args.force_apply:
        cfg["mode"] = "apply"
    elif args.force_preview:
        cfg["mode"] = "preview"

    mode = cfg.get("mode", "preview").lower()
    if mode not in ("preview", "apply"):
        logger.error(f"Mode không hợp lệ: {mode}. Phải là 'preview' hoặc 'apply'.")
        sys.exit(1)
    write_output = (mode == "apply")

    logger.info(f"Config         : {config_path}")
    logger.info(f"Mode           : {mode.upper()}")

    # Resolve paths
    input_dir = (PROJECT_ROOT / cfg["input_dir"]).resolve()
    output_dir = (PROJECT_ROOT / cfg["output_dir"]).resolve()
    mapping_path = (PREPARE_DIR / cfg["english_mapping_file"]).resolve()

    logger.info(f"Input dir      : {input_dir}")
    logger.info(f"Output dir     : {output_dir}")
    logger.info(f"Mapping file   : {mapping_path}")

    if not input_dir.exists():
        logger.error(f"Input dir không tồn tại: {input_dir}")
        sys.exit(1)

    # Load mapping
    try:
        delete_keywords, mapping = parse_english_mapping(mapping_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        sys.exit(1)

    # Bổ sung extra delete keywords từ config
    extra = cfg.get("extra_delete_keywords") or []
    for kw in extra:
        kw_l = kw.lower().strip()
        if kw_l and kw_l not in delete_keywords:
            delete_keywords.append(kw_l)

    logger.info(f"Tier 1 keywords ({len(delete_keywords)}): {delete_keywords}")
    logger.info(f"Tier 2 mappings : {len(mapping)} entries")
    if len(mapping) <= 30:
        for k, v in mapping.items():
            logger.info(f"   '{k}' -> '{v}'")
    logger.info("-" * 60)

    # Process từng split
    total_stats = CleanStats()
    for split, in_name in cfg["input_files"].items():
        in_path = input_dir / in_name
        out_name = cfg["output_files"][split]
        out_path = output_dir / out_name
        logger.info(f"\n>>> Xử lý split: {split.upper()}")
        s = process_file(
            in_path, out_path,
            delete_keywords, mapping, cfg,
            logger, write_output,
        )
        total_stats.total += s.total
        total_stats.kept += s.kept
        total_stats.deleted_tier1 += s.deleted_tier1
        total_stats.deleted_tier3 += s.deleted_tier3
        total_stats.deleted_empty += s.deleted_empty
        total_stats.modified += s.modified
        for w, c in s.unknown_english_words.items():
            total_stats.unknown_english_words[w] = (
                total_stats.unknown_english_words.get(w, 0) + c
            )

    # Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT")
    logger.info("=" * 60)
    logger.info(f"  Tổng input         : {total_stats.total}")
    logger.info(f"  Kept (output)      : {total_stats.kept}")
    logger.info(f"  Modified           : {total_stats.modified}")
    logger.info(f"  Deleted Tier 1     : {total_stats.deleted_tier1}")
    logger.info(f"  Deleted Tier 3     : {total_stats.deleted_tier3}")
    logger.info(f"  Deleted empty      : {total_stats.deleted_empty}")

    if total_stats.unknown_english_words:
        sorted_unknown = sorted(
            total_stats.unknown_english_words.items(),
            key=lambda x: -x[1],
        )
        max_log = cfg.get("max_log_unknown_words", 30)
        logger.info(
            f"\n  Tier 3 đã remove {sum(total_stats.unknown_english_words.values())} "
            f"từ tiếng Anh không có trong mapping "
            f"({len(total_stats.unknown_english_words)} unique). "
            f"Top {min(max_log, len(sorted_unknown))}:"
        )
        for w, c in sorted_unknown[:max_log]:
            logger.info(f"    '{w}' x{c}")
        logger.info(
            "\n  GỢI Ý: Nếu thấy từ Việt hợp lệ bị bắt nhầm, "
            "hãy thêm nó vào english_mapping.txt với mapping về CHÍNH NÓ "
            "(vd: 'banh, bánh') để bypass Tier 3."
        )

    logger.info("\nBước tiếp theo:")
    if not write_output:
        logger.info("  - Hiện đang ở PREVIEW MODE -> KHÔNG ghi file output.")
        logger.info("  - Nếu kết quả OK, đổi 'mode: \"apply\"' trong config "
                    "rồi chạy lại.")
        logger.info("  - Hoặc thêm flag: --force-apply")
    else:
        logger.info("  - File output đã ghi. Tiếp theo sửa NHẸ A1 "
                    "(step1_rephonemize_lite.py) để đọc input từ:")
        logger.info(f"      {output_dir}/filelist_train_clean.txt")
        logger.info(f"      {output_dir}/filelist_val_clean.txt")
        logger.info("    Sau đó chạy lại A1 -> A2.")


if __name__ == "__main__":
    main()
