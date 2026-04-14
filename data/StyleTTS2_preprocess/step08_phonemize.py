"""
=============================================================
  BƯỚC 8: G2P (Grapheme-to-Phoneme) — Chuyển đổi Âm vị
=============================================================
Mục tiêu: Chuyển đổi văn bản tiếng Việt sang chuỗi âm vị (phoneme)
          để tương thích với bộ từ vựng của StyleTTS2-vi.
=============================================================
Chạy lệnh: python -X utf8 .\step08_phonemize.py
"""

import os
import sys
import logging
import re
from pathlib import Path
from tqdm import tqdm

# ==========================================
# KHẮC PHỤC LỖI "CHARMAP CODEC" TRÊN WINDOWS
# ==========================================
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass
    os.environ["PYTHONIOENCODING"] = "utf-8"
    os.environ["PYTHONUTF8"] = "1"

# ==========================================
# KHẮC PHỤC LỖI "WinError 193" (Bypass Linux Binary)
# ==========================================
import vinorm

# 1. Tạo hàm giả (Mock Function) để thay thế module Linux
def mock_tts_norm(text, *args, **kwargs):
    # Bỏ qua file thực thi Linux, chỉ chuyển text về chữ thường
    # vì bộ từ điển của viphoneme map tốt nhất với chữ thường.
    return str(text).lower().strip()

# 2. Ghi đè (Monkey-patch) trực tiếp vào thư viện vinorm trên bộ nhớ
vinorm.TTSnorm = mock_tts_norm
vinorm.TTSrawUpper = lambda t, *a, **k: str(t).strip()

import viphoneme
# 3. Ghi đè tiếp vào không gian tên của viphoneme 
# (do thư viện này thiết kế import chồng chéo)
viphoneme.TTSnorm = mock_tts_norm

# 4. Bây giờ mới import hàm G2P cốt lõi một cách an toàn
from viphoneme import vi2IPA_split

# ==========================================

def setup_logging(log_path: str) -> logging.Logger:
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_path, encoding="utf-8", mode="w"),
        ],
    )
    return logging.getLogger("step08")

def text_to_phoneme(text: str) -> str:
    """
    Sử dụng vi2IPA_split với delimit=" " để tách rời từng âm vị.
    """
    try:
        phonemes = vi2IPA_split(text, " ")
        phonemes = re.sub(r'\s+', ' ', phonemes).strip()
        return phonemes
    except Exception as e:
        return f"[ERROR] {str(e)}"

def process_filelist(input_path: Path, output_path: Path, logger: logging.Logger):
    if not input_path.exists():
        logger.warning(f"Bỏ qua (không tìm thấy): {input_path}")
        return

    # Đọc file với chuẩn UTF-8
    with open(input_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    logger.info(f"Đã tải {len(lines)} records từ {input_path.name}")
    processed_records = []
    error_count = 0

    for line in tqdm(lines, desc=f"Xử lý {input_path.name}", ncols=80):
        line = line.strip()
        if not line:
            continue
            
        parts = line.split("|")
        
        if len(parts) == 2:
            wav_path = parts[0]
            text = parts[1]
            phoneme_str = text_to_phoneme(text)
            
            if phoneme_str.startswith("[ERROR]"):
                logger.warning(f"Lỗi G2P: {text} -> {phoneme_str}")
                error_count += 1
                continue
                
            processed_records.append(f"{wav_path}|{phoneme_str}")
            
        elif len(parts) == 1:
            text = parts[0]
            phoneme_str = text_to_phoneme(text)
            
            if phoneme_str.startswith("[ERROR]"):
                logger.warning(f"Lỗi G2P (OOD Text): {text} -> {phoneme_str}")
                error_count += 1
                continue
                
            processed_records.append(phoneme_str)
        else:
            error_count += 1

    # Ghi file với chuẩn UTF-8
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(processed_records) + "\n")

    logger.info(f"Hoàn tất {input_path.name}! OK: {len(processed_records)} | Lỗi: {error_count}\n")

def main():
    work_dir = Path("./workdir")
    output_dir = Path("./output_dataset")
    log_path = str(work_dir / "logs" / "step08_phonemize.log")
    
    logger = setup_logging(log_path)
    logger.info("=" * 50)
    logger.info("  BƯỚC 8: CHUYỂN ĐỔI ÂM VỊ (G2P) - WINDOWS 11")
    logger.info("=" * 50)

    target_files = [
        "filelist_train.txt", 
        "filelist_val.txt", 
        "OOD_texts.txt"
    ]

    for filename in target_files:
        in_path = output_dir / filename
        out_path = output_dir / filename.replace(".txt", "_phoneme.txt")
        process_filelist(in_path, out_path, logger)

if __name__ == "__main__":
    main()