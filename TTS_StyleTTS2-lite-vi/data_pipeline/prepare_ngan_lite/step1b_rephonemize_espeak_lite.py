"""
=============================================================
  STEP 1B (A1B): RE-PHONEMIZE NGAN VOICE BY ESPEAK-NG
=============================================================
Mục tiêu:
  Thay thế step1_rephonemize_lite.py (viphoneme version) sau khi
  A3.2 zero-shot test confirm rằng pretrained StyleTTS2-lite-vi đọc
  TỐT phoneme từ espeak-ng và KÉM với phoneme từ viphoneme (bị
  "tách chữ" vì replace '_' -> space tạo ra các token đơn lẻ).

  Espeak-ng cho output IPA liền mạch theo từng từ, match với cách
  inference.py của lite-vi phonemize -> fine-tune sẽ converge nhanh
  và chất lượng tốt hơn.

Input:
  - TTS_StyleTTS2-lite-vi/output/filelist_train_clean.txt
  - TTS_StyleTTS2-lite-vi/output/filelist_val_clean.txt
  (output từ step0_clean_text.py)

Output (OVERWRITE file cũ từ step1 viphoneme):
  - TTS_StyleTTS2-lite-vi/output/ngan_train_phoneme_raw.txt
  - TTS_StyleTTS2-lite-vi/output/ngan_val_phoneme_raw.txt

Sau bước này: chạy LẠI step2_make_filelist_lite.py (A2) như cũ
KHÔNG cần sửa gì. Kỳ vọng vocab_fail = 0 vì lite-vi train trên
chính format espeak này.

Dependencies (cài 1 lần trên Windows local):
    pip install phonemizer espeakng-loader

Cách chạy (từ root TTS_StyleTTS2-lite-vi/):
    python -X utf8 data_pipeline/prepare_ngan_lite/step1b_rephonemize_espeak_lite.py
=============================================================
"""

import os
import sys
import re
import logging
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

# ============================================================
# SECTION 1: WINDOWS UTF-8 FIX
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
# SECTION 2: ESPEAK-NG SETUP (Windows fix)
# ----------------------------------------------------------------
# Ưu tiên native espeak-ng installer (ổn nhất),
# fallback sang espeakng_loader nếu không có.
# ============================================================
def setup_espeak() -> None:
    """Setup espeak-ng binary + data path cho mọi nền tảng."""
    if sys.platform != "win32":
        # Linux/Mac: phonemizer tự tìm espeak-ng system-wide
        return

    # --- Windows: thử native install trước ---
    program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
    native_lib  = Path(program_files) / "eSpeak NG" / "libespeak-ng.dll"
    native_exe  = Path(program_files) / "eSpeak NG" / "espeak-ng.exe"
    native_data = Path(program_files) / "eSpeak NG" / "espeak-ng-data"

    if native_lib.is_file() and native_data.is_dir():
        # Phonemizer đọc các env var này khi import
        os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(native_lib)
        os.environ["PHONEMIZER_ESPEAK_PATH"]    = str(native_exe)
        os.environ["ESPEAK_DATA_PATH"]          = str(native_data)
        # Verbose chỉ để debug
        print(f"[espeak] Using native install: {native_lib.parent}")
        return

    # --- Fallback: espeakng_loader (KHÔNG ổn định trên Windows) ---
    try:
        from phonemizer.backend.espeak.wrapper import EspeakWrapper
        import espeakng_loader

        lib_path  = espeakng_loader.get_library_path()
        data_path = espeakng_loader.get_data_path()

        EspeakWrapper.set_library(lib_path)
        # set_data_path chỉ tồn tại ở phonemizer-fork hoặc phonemizer >= 3.3
        if hasattr(EspeakWrapper, "set_data_path"):
            EspeakWrapper.set_data_path(data_path)
        else:
            os.environ["ESPEAK_DATA_PATH"] = data_path

        print(f"[espeak] Using espeakng_loader fallback: {lib_path}")
    except ImportError as e:
        raise ImportError(
            "Không tìm thấy espeak-ng native VÀ thiếu phonemizer/espeakng-loader.\n"
            "Cách 1 (khuyên dùng): cài espeak-ng-X64.msi từ "
            "https://github.com/espeak-ng/espeak-ng/releases/latest\n"
            "Cách 2: pip install phonemizer espeakng-loader\n"
            f"Chi tiết: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Setup espeak-ng thất bại: {e}")

# ============================================================
# SECTION 3: PATHS
# ----------------------------------------------------------------
# Cấu trúc:
#   TTS_StyleTTS2-lite-vi/
#   ├── data_pipeline/prepare_ngan_lite/
#   │   └── step1b_rephonemize_espeak_lite.py    <-- FILE NÀY
#   ├── output/
#   │   ├── filelist_train_clean.txt              <-- INPUT (từ step0)
#   │   ├── filelist_val_clean.txt
#   │   ├── ngan_train_phoneme_raw.txt            <-- OUTPUT (overwrite)
#   │   └── ngan_val_phoneme_raw.txt
#   └── logs/
# ============================================================
SCRIPT_PATH = Path(__file__).resolve()
PREPARE_DIR = SCRIPT_PATH.parent
DATA_PIPELINE_DIR = PREPARE_DIR.parent
PROJECT_ROOT = DATA_PIPELINE_DIR.parent           # TTS_StyleTTS2-lite-vi/

INPUT_DIR = PROJECT_ROOT / "output"               # filelist_*_clean.txt
OUTPUT_DIR = PROJECT_ROOT / "output"              # *_phoneme_raw.txt (overwrite)
LOG_DIR = PROJECT_ROOT / "logs"

INPUT_FILES = {
    "train": INPUT_DIR / "filelist_train_clean.txt",
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
    log_path = LOG_DIR / "step1b_rephonemize_espeak_lite.log"

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
    return logging.getLogger("step1b_espeak")


# ============================================================
# SECTION 5: ESPEAK PHONEMIZER
# ----------------------------------------------------------------
# Settings GIỐNG HỆT inference.py của lite-vi:
#   EspeakBackend(
#       language=lang,
#       preserve_punctuation=True,
#       with_stress=True,
#       language_switch="remove-flags",
#   )
#
# Sau phonemize: re.sub(r'\s+', ' ', sent).strip()
#
# Đây là logic chính xác mà pretrained lite-vi expect.
# ============================================================
class EspeakPhonemizer:
    """Wrapper khởi tạo backend 1 lần, dùng cho cả batch."""

    def __init__(self, language: str = "vi"):
        from phonemizer.backend import EspeakBackend
        self.backend = EspeakBackend(
            language=language,
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
        )

    def phonemize_single(self, text: str) -> str:
        """Phonemize 1 câu. Trả về chuỗi phoneme đã normalize."""
        out = self.backend.phonemize([text])[0]
        out = re.sub(r"\s+", " ", out).strip()
        return out

    def phonemize_batch(self, texts: List[str]) -> List[str]:
        """
        Phonemize 1 batch (NHANH hơn 5-10x so với loop từng câu).
        Empty input -> empty output, không crash.
        """
        if not texts:
            return []
        outs = self.backend.phonemize(texts)
        return [re.sub(r"\s+", " ", o).strip() for o in outs]


# ============================================================
# SECTION 6: FILE PROCESSING
# ============================================================
def process_filelist(
    input_path: Path,
    output_path: Path,
    phonemizer: EspeakPhonemizer,
    batch_size: int,
    logger: logging.Logger,
) -> dict:
    """
    Xử lý 1 file filelist.

    Định dạng input mỗi dòng:  wav_path | text  (text đã clean)
    Định dạng output mỗi dòng: wav_path | espeak_phoneme

    wav_path được GIỮ NGUYÊN — A2 chịu trách nhiệm normalize path.
    """
    if not input_path.exists():
        logger.error(f"Không tìm thấy file input: {input_path}")
        return {"ok": 0, "errors": 0, "skipped": 0}

    with open(input_path, "r", encoding="utf-8") as f:
        raw_lines = f.readlines()

    logger.info(f"Đã load {len(raw_lines)} dòng từ {input_path.name}")

    # ===== Parse từng dòng =====
    parsed: List[Tuple[int, str, str]] = []  # (line_num, wav_path, text)
    skipped_count = 0
    for line_num, line in enumerate(raw_lines, start=1):
        line = line.strip()
        if not line:
            skipped_count += 1
            continue
        parts = line.split("|", 1)
        if len(parts) != 2:
            logger.warning(
                f"  Dòng {line_num}: format sai (cần 2 cột). Bỏ qua: {line[:80]}"
            )
            skipped_count += 1
            continue
        wav_path, text = parts[0].strip(), parts[1].strip()
        if not wav_path or not text:
            skipped_count += 1
            continue
        parsed.append((line_num, wav_path, text))

    if not parsed:
        logger.error("Không có dòng nào hợp lệ để phonemize.")
        return {"ok": 0, "errors": 0, "skipped": skipped_count}

    # ===== Batch phonemize =====
    processed_records: List[str] = []
    error_count = 0
    sample_logged = False

    # Chia thành batches để có progress bar & resilient với crash
    n_batches = (len(parsed) + batch_size - 1) // batch_size
    for b_idx in tqdm(range(n_batches),
                      desc=f"Espeak {input_path.name}",
                      ncols=90):
        start = b_idx * batch_size
        end = min(start + batch_size, len(parsed))
        batch = parsed[start:end]
        batch_texts = [t for _, _, t in batch]

        # Phonemize cả batch — nếu fail thì fallback từng dòng
        try:
            batch_phonemes = phonemizer.phonemize_batch(batch_texts)
        except Exception as e:
            logger.warning(
                f"  Batch {b_idx} fail cả khối ({type(e).__name__}: {e})."
                f" Fallback từng dòng..."
            )
            batch_phonemes = []
            for txt in batch_texts:
                try:
                    batch_phonemes.append(phonemizer.phonemize_single(txt))
                except Exception as e2:
                    batch_phonemes.append(f"[ERROR] {type(e2).__name__}: {e2}")

        # Đóng gói output
        for (line_num, wav_path, text), phn in zip(batch, batch_phonemes):
            if phn.startswith("[ERROR]") or not phn:
                error_count += 1
                if error_count <= 5:
                    logger.warning(
                        f"  Dòng {line_num} phonemize fail: {phn or '<empty>'}\n"
                        f"    Text: {text[:100]}"
                    )
                continue

            processed_records.append(f"{wav_path}|{phn}")

            if not sample_logged:
                logger.info(
                    f"\n  [SAMPLE]\n"
                    f"    text gốc : {text}\n"
                    f"    phoneme  : {phn}\n"
                    f"    wav_path : {wav_path}"
                )
                sample_logged = True

    # ===== Ghi output =====
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Cảnh báo nếu overwrite file cũ
    if output_path.exists():
        file_size = output_path.stat().st_size
        logger.warning(
            f"  ! Sắp OVERWRITE file cũ: {output_path}\n"
            f"    File cũ size: {file_size:,} bytes (có thể là output viphoneme)"
        )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_records))
        if processed_records:
            f.write("\n")

    logger.info(
        f"\n  Hoàn tất: {input_path.name}\n"
        f"    OK     : {len(processed_records)} dòng\n"
        f"    Errors : {error_count} dòng\n"
        f"    Skipped: {skipped_count} dòng (rỗng / format sai)\n"
        f"    -> Đã ghi: {output_path}"
    )

    return {
        "ok": len(processed_records),
        "errors": error_count,
        "skipped": skipped_count,
    }


# ============================================================
# SECTION 7: MAIN
# ============================================================
def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Số câu phonemize cùng lúc (default 64). Tăng nếu RAM dư.",
    )
    parser.add_argument(
        "--language",
        type=str,
        default="vi",
        help="Espeak language code (default 'vi' cho Vietnamese).",
    )
    args = parser.parse_args()

    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("  STEP 1B: RE-PHONEMIZE BY ESPEAK-NG")
    logger.info("=" * 60)
    logger.info(f"Project root   : {PROJECT_ROOT}")
    logger.info(f"Input dir      : {INPUT_DIR}")
    logger.info(f"Output dir     : {OUTPUT_DIR}")
    logger.info(f"Language       : {args.language}")
    logger.info(f"Batch size     : {args.batch_size}")
    logger.info("-" * 60)

    # Sanity check INPUT_DIR
    if not INPUT_DIR.exists():
        logger.error(f"INPUT_DIR không tồn tại: {INPUT_DIR}")
        logger.error("Hãy chạy step0_clean_text.py (mode apply) trước.")
        sys.exit(1)

    for split, in_path in INPUT_FILES.items():
        if not in_path.exists():
            logger.error(f"Thiếu file input {split}: {in_path}")
            logger.error(
                "Đảm bảo step0_clean_text.py đã chạy ở mode 'apply' để "
                "sinh ra filelist_*_clean.txt."
            )
            sys.exit(1)

    # Setup espeak-ng binary
    logger.info("Đang setup espeak-ng...")
    try:
        setup_espeak()
    except Exception as e:
        logger.error(str(e))
        sys.exit(1)

    # Init phonemizer
    logger.info("Đang khởi tạo EspeakBackend...")
    try:
        phonemizer = EspeakPhonemizer(language=args.language)
    except Exception as e:
        logger.error(
            f"Khởi tạo phonemizer thất bại: {type(e).__name__}: {e}\n"
            "Có thể do:\n"
            "  - espeak-ng binary không tìm thấy (Windows: cài espeakng-loader)\n"
            "  - language code sai (Việt = 'vi', không phải 'vie' hay 'vn')"
        )
        sys.exit(1)

    # Sanity test với 1 câu
    logger.info("Test phonemize với câu mẫu...")
    test_text = "Đồng hồ điểm đúng mười hai giờ đêm."
    try:
        test_phn = phonemizer.phonemize_single(test_text)
        if not test_phn:
            raise RuntimeError("Phoneme output rỗng")
        logger.info(
            f"  Test OK:\n"
            f"    Input  : {test_text}\n"
            f"    Output : {test_phn}"
        )
    except Exception as e:
        logger.error(f"Test phonemize thất bại: {type(e).__name__}: {e}")
        sys.exit(1)
    logger.info("-" * 60)

    # Process từng split
    total_stats = {"ok": 0, "errors": 0, "skipped": 0}
    for split, in_path in INPUT_FILES.items():
        logger.info(f"\n>>> Xử lý split: {split.upper()}")
        out_path = OUTPUT_FILES[split]
        stats = process_filelist(
            in_path, out_path, phonemizer,
            batch_size=args.batch_size,
            logger=logger,
        )
        for k in total_stats:
            total_stats[k] += stats.get(k, 0)

    # Tổng kết
    logger.info("\n" + "=" * 60)
    logger.info("TỔNG KẾT")
    logger.info("=" * 60)
    logger.info(f"  Tổng OK     : {total_stats['ok']}")
    logger.info(f"  Tổng errors : {total_stats['errors']}")
    logger.info(f"  Tổng skipped: {total_stats['skipped']}")
    logger.info("")
    logger.info("Bước tiếp theo:")
    logger.info("  - Chạy LẠI step2_make_filelist_lite.py (A2) KHÔNG cần sửa gì.")
    logger.info("    Kỳ vọng: vocab_fail = 0 (vì lite-vi train chính trên espeak).")
    logger.info("  - Lệnh:")
    logger.info(
        "    python -X utf8 data_pipeline/prepare_ngan_lite/"
        "step2_make_filelist_lite.py"
    )


if __name__ == "__main__":
    main()
