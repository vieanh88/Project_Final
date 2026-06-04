"""
=============================================================
  STEP 1 (A1): RE-PHONEMIZE NGAN VOICE FOR StyleTTS2-lite-vi
  SỬ DỤNG VIPHONEME (VIPHONE_VI2IPA_SPLIT) + VINORM MONKEY-PATCH
=============================================================
Mục tiêu:
  - Đọc text gốc tiếng Việt từ pipeline cũ (StyleTTS2_preprocess).
  - Phonemize bằng viphoneme.vi2IPA_split (giữ nguyên logic step08
    để tận dụng infra Windows + vinorm monkey-patch đã verify).
  - Replace ký tự '_' (separator phoneme nội bộ của viphoneme) bằng
    space — vì vocab 189 symbols của StyleTTS2-lite-vi KHÔNG có '_'.
  - Output: ngan_train_phoneme_raw.txt + ngan_val_phoneme_raw.txt
            đặt trong TTS_StyleTTS2-lite-vi/output/.
  - File "_raw" này là intermediate. File A2 sẽ validate vocab,
    normalize path và split thành format final cho training.

Cách chạy (từ root TTS_StyleTTS2-lite-vi/):
    python -X utf8 data_pipeline/prepare_ngan_lite/step1_rephonemize_viphoneme_lite.py

Hoặc từ thư mục chứa file:
    python -X utf8 step1_rephonemize_viphoneme_lite.py
=============================================================
"""

import os
import sys
import logging
import re
from pathlib import Path
from tqdm import tqdm

# ============================================================
# SECTION 1: WINDOWS UTF-8 FIX (giữ nguyên từ step08 cũ)
# ============================================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# ============================================================
# SECTION 2: VINORM/VIPHONEME MONKEY-PATCH
# ----------------------------------------------------------------
# Bypass binary Linux của vinorm (gây WinError 193 trên Windows).
# Logic: thay TTSnorm bằng mock function, chỉ lower + strip text.
# Phải patch TRƯỚC khi import vi2IPA_split, vì viphoneme import
# vinorm ở module-level.
# ============================================================
import vinorm

def _mock_tts_norm(text, *args, **kwargs):
    return str(text).lower().strip()

vinorm.TTSnorm = _mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

import viphoneme
viphoneme.TTSnorm = _mock_tts_norm  # patch lần 2 do import chồng chéo

from viphoneme import vi2IPA_split

# ============================================================
# SECTION 3: PATHS — CẤU HÌNH ĐƯỜNG DẪN
# ----------------------------------------------------------------
# Phát hiện vị trí script và resolve các path tuyệt đối.
# Cấu trúc dự án (theo confirm của user):
#
#   HUST_Project/Project_Final/
#   ├── data/
#   │   └── StyleTTS2_preprocess/
#   │       └── output_dataset/
#   │           ├── filelist_train.txt   <-- INPUT
#   │           └── filelist_val.txt     <-- INPUT
#   └── TTS_StyleTTS2-lite-vi/
#       ├── data_pipeline/
#       │   └── prepare_ngan_lite/
#       │       └── step1_rephonemize_viphoneme_lite.py  <-- FILE NÀY
#       └── output/
#           ├── ngan_train_phoneme_raw.txt     <-- OUTPUT
#           └── ngan_val_phoneme_raw.txt       <-- OUTPUT
#
# Đổi đường dẫn để lấy input từ output của step0 (filelist_train_clean.txt, filelist_val_clean.txt)
# D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\output\filelist_train_clean.txt
# D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\output\filelist_val_clean.txt
# ============================================================
SCRIPT_PATH = Path(__file__).resolve()
PREPARE_DIR = SCRIPT_PATH.parent                       # .../prepare_ngan_lite/
DATA_PIPELINE_DIR = PREPARE_DIR.parent                 # .../data_pipeline/
PROJECT_ROOT = DATA_PIPELINE_DIR.parent                # .../TTS_StyleTTS2-lite-vi/
PROJECT_FINAL_ROOT = PROJECT_ROOT.parent               # .../Project_Final/

INPUT_DIR = PROJECT_ROOT / "output"          # ← đọc từ output của step0
OUTPUT_DIR = PROJECT_ROOT / "output"
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_FILES = {
    "train": INPUT_DIR / "filelist_train_clean.txt",   # ← file cleaned
    "val":   INPUT_DIR / "filelist_val_clean.txt",
}
OUTPUT_FILES = {
    "train": OUTPUT_DIR / "ngan_train_phoneme_raw.txt",
    "val":   OUTPUT_DIR / "ngan_val_phoneme_raw.txt",
}

# ============================================================
# SECTION 4: LOGGING
# ============================================================
def setup_logging() -> logging.Logger:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / "step1_rephonemize_lite.log"

    # Reset handlers nếu chạy lại trong cùng interpreter
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
    return logging.getLogger("step1_rephonemize_lite")


# ============================================================
# SECTION 5: CORE PHONEMIZATION
# ============================================================
def text_to_phoneme_lite(text: str) -> str:
    """
    Chuyển text tiếng Việt -> chuỗi phoneme tương thích StyleTTS2-lite-vi.

    Pipeline:
      1. vi2IPA_split(text, " ") -> "k_ɤ̆_j_1 ɣ_i_2 ..."
         (phoneme trong cùng âm tiết nối bằng '_', âm tiết cách nhau bằng ' ')
      2. Replace '_' -> ' ' để mỗi phoneme là một token độc lập:
         "k ɤ̆ j 1 ɣ i 2 ..."
      3. Collapse multiple whitespace -> single space.

    Lý do replace '_':
      Vocab StyleTTS2-lite-vi (189 symbols, từ Configs/config.yaml) gồm:
        pad + punctuation + letters + letters_ipa + extend
      Trong đó NONE chứa ký tự '_'. Nếu giữ '_' thì TextCleaner sẽ
      silently skip (KeyError -> continue), làm sai lệch alignment.
    """
    try:
        phonemes = vi2IPA_split(text, " ")
        if not phonemes:
            return "[ERROR] Empty phoneme output"
        # Replace separator nội bộ '_' -> space
        phonemes = phonemes.replace("_", " ")
        # Normalize whitespace
        phonemes = re.sub(r"\s+", " ", phonemes).strip()
        if not phonemes:
            return "[ERROR] Phonemes became empty after normalize"
        return phonemes
    except Exception as e:
        return f"[ERROR] {type(e).__name__}: {str(e)}"


# ============================================================
# SECTION 6: FILE PROCESSING
# ============================================================
def process_filelist(
    input_path: Path,
    output_path: Path,
    logger: logging.Logger,
) -> dict:
    """
    Xử lý 1 file filelist.

    Định dạng input mỗi dòng:  wav_path|text
    Định dạng output mỗi dòng: wav_path|phoneme

    wav_path được giữ NGUYÊN (kể cả backslash '\\' của Windows).
    Bước A2 sẽ chịu trách nhiệm normalize path khi build filelist final.
    """
    if not input_path.exists():
        logger.error(f"Không tìm thấy file input: {input_path}")
        return {"ok": 0, "errors": 0, "skipped": 0}

    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đã load {len(lines)} dòng từ {input_path.name}")

    processed = []
    error_count = 0
    skipped_count = 0
    sample_logged = False  # log 1 ví dụ thực tế cho user verify

    for line_num, line in enumerate(
        tqdm(lines, desc=f"Phonemize {input_path.name}", ncols=90), start=1
    ):
        line = line.strip()
        if not line:
            skipped_count += 1
            continue

        parts = line.split("|")
        if len(parts) != 2:
            logger.warning(
                f"  Dòng {line_num}: định dạng sai (cần 2 cột, có {len(parts)}). "
                f"Bỏ qua: {line[:80]}..."
            )
            skipped_count += 1
            continue

        wav_path, text = parts[0].strip(), parts[1].strip()

        if not wav_path or not text:
            logger.warning(f"  Dòng {line_num}: wav_path hoặc text rỗng. Bỏ qua.")
            skipped_count += 1
            continue

        phoneme = text_to_phoneme_lite(text)

        if phoneme.startswith("[ERROR]"):
            error_count += 1
            if error_count <= 5:  # chỉ log 5 lỗi đầu để tránh spam
                logger.warning(
                    f"  Dòng {line_num} G2P thất bại: {phoneme}\n"
                    f"    Text gốc: {text[:100]}"
                )
            continue

        processed.append(f"{wav_path}|{phoneme}")

        # Log 1 ví dụ đầu tiên thành công để user kiểm tra format
        if not sample_logged:
            logger.info(
                f"  [SAMPLE] Input  text : {text}\n"
                f"           Output phn : {phoneme}\n"
                f"           wav_path   : {wav_path}"
            )
            sample_logged = True

    # Đảm bảo thư mục output tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed))
        if processed:
            f.write("\n")

    logger.info(
        f"  Hoàn tất: {input_path.name}\n"
        f"    OK     : {len(processed)} dòng\n"
        f"    Lỗi G2P: {error_count} dòng\n"
        f"    Skipped: {skipped_count} dòng (rỗng/sai format)\n"
        f"    -> Đã ghi: {output_path}"
    )

    return {
        "ok": len(processed),
        "errors": error_count,
        "skipped": skipped_count,
    }


# ============================================================
# SECTION 7: MAIN
# ============================================================
def main():
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("  STEP 1: RE-PHONEMIZE NGAN VOICE (StyleTTS2-lite-vi)")
    logger.info("=" * 60)
    logger.info(f"Project root         : {PROJECT_ROOT}")
    logger.info(f"Input directory      : {INPUT_DIR}")
    logger.info(f"Output directory     : {OUTPUT_DIR}")
    logger.info(f"Log directory        : {LOG_DIR}")
    logger.info("-" * 60)

    # Sanity-check INPUT_DIR
    if not INPUT_DIR.exists():
        logger.error(
            f"INPUT_DIR không tồn tại: {INPUT_DIR}\n"
            f"  Hãy kiểm tra:\n"
            f"  - Bạn đã copy data từ pipeline cũ vào "
            f"data/StyleTTS2_preprocess/ chưa?\n"
            f"  - File này đang đặt đúng vị trí "
            f"TTS_StyleTTS2-lite-vi/data_pipeline/prepare_ngan_lite/ chưa?"
        )
        sys.exit(1)

    # Quick test viphoneme trước khi xử lý hàng loạt
    logger.info("Đang test viphoneme với câu mẫu...")
    test_text = "Đồng hồ điểm đúng mười hai giờ đêm."
    test_phn = text_to_phoneme_lite(test_text)
    if test_phn.startswith("[ERROR]"):
        logger.error(f"viphoneme test thất bại: {test_phn}")
        logger.error("Hãy kiểm tra cài đặt vinorm + viphoneme.")
        sys.exit(1)
    logger.info(f"  Test OK: '{test_text}' -> '{test_phn}'")
    logger.info("-" * 60)

    # Process từng filelist
    total_stats = {"ok": 0, "errors": 0, "skipped": 0}
    for split_name, in_path in INPUT_FILES.items():
        logger.info(f"\n>>> Xử lý split: {split_name.upper()}")
        out_path = OUTPUT_FILES[split_name]
        stats = process_filelist(in_path, out_path, logger)
        for k in total_stats:
            total_stats[k] += stats[k]

    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT")
    logger.info("=" * 60)
    logger.info(f"  Tổng dòng OK     : {total_stats['ok']}")
    logger.info(f"  Tổng lỗi G2P     : {total_stats['errors']}")
    logger.info(f"  Tổng skipped     : {total_stats['skipped']}")
    logger.info("\nBước tiếp theo: chạy file A2 (step2_make_filelist_lite.py)")
    logger.info("để validate vocab, normalize path và build filelist final.")

if __name__ == "__main__":
    main()