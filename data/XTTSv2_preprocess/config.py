"""
=============================================================
  CONFIG - Audio Pipeline cho Giọng Nguyễn Ngọc Ngạn
  Mục tiêu: Tạo dataset sạch cho Fine-tuning XTTS v2
=============================================================
Chỉnh sửa các thông số tại đây trước khi chạy pipeline.
"""

import os
from dotenv import load_dotenv
load_dotenv()  # Load biến môi trường từ .env
# ------------------------------------------------------------------ #
#  ĐƯỜNG DẪN
# ------------------------------------------------------------------ #
# Thư mục chứa các file audio gốc (MP3 / WAV / FLAC)
INPUT_DIR = "./raw_audio"

# Thư mục lưu kết quả cuối cùng (dry voice của Ngạn, chuẩn XTTS v2)
OUTPUT_DIR = "./output_ngan_voice"

# Thư mục trung gian (có thể xóa sau khi xong)
WORK_DIR = "./workdir"

# HuggingFace token (bắt buộc cho pyannote)
# Khuyến nghị: đặt biến môi trường HF_API_KEY thay vì hardcode
HF_TOKEN = os.getenv("HF_API_KEY")

# ------------------------------------------------------------------ #
#  BƯỚC 1: VOCAL SEPARATION (Demucs)
# ------------------------------------------------------------------ #
# Model Demucs: "htdemucs_ft" cho chất lượng tốt nhất
# Hoặc "htdemucs" nếu RAM/VRAM bị giới hạn
DEMUCS_MODEL = "htdemucs_ft"

# Chạy Demucs trên CPU nếu VRAM < 4GB lúc chạy bước này
# True = CPU (chậm ~3-5x nhưng an toàn VRAM), False = GPU
DEMUCS_CPU_ONLY = False

# ------------------------------------------------------------------ #
#  BƯỚC 2: DIARIZATION (pyannote-audio)
# ------------------------------------------------------------------ #
# Model pyannote - cần accept terms tại:
# https://huggingface.co/pyannote/speaker-diarization-3.1
# https://huggingface.co/pyannote/segmentation-3.0
PYANNOTE_MODEL = "pyannote/speaker-diarization-3.1"

# Số speaker tối đa ước lượng trong 1 file audio
# Truyện Ngọc Ngạn thường: 1 narrator + 1-2 nhân vật + SFX → đặt 3
DIARIZATION_MAX_SPEAKERS = 3

# Merge các segment cùng speaker gần nhau (giây)
# Nếu 2 đoạn cùng speaker cách nhau < giá trị này → gộp lại
MERGE_GAP_SECONDS = 0.3

# ------------------------------------------------------------------ #
#  BƯỚC 3: SPEAKER VERIFICATION
# ------------------------------------------------------------------ #
# Threshold cosine similarity để kết luận "đây là giọng Ngạn"
# Range [0, 1] — khuyến nghị 0.75 (chặt). Hạ xuống 0.65 nếu bỏ sót nhiều.
SPEAKER_SIMILARITY_THRESHOLD = 0.65

# Số segment tốt nhất dùng để xây reference embedding (bootstrap)
# Lấy từ dominant speaker (người nói lâu nhất)
N_REFERENCE_SEGMENTS = 20

# Thời lượng tối thiểu (giây) của segment dùng làm reference
MIN_REFERENCE_DURATION = 5.0

# ------------------------------------------------------------------ #
#  BƯỚC 4: LỌC CHẤT LƯỢNG
# ------------------------------------------------------------------ #

# --- Thời lượng segment ---
# XTTSv2 yêu cầu: tối thiểu 2s - tối đa 12s mỗi đoạn
MIN_SEGMENT_DURATION = 2.0   # giây
MAX_SEGMENT_DURATION = 12.0  # giây

# --- SNR (Signal-to-Noise Ratio) ---
# Đoạn nào SNR < ngưỡng này sẽ bị loại (còn nhiều nhạc nền, noise)
MIN_SNR_DB = 10.0

# --- Phát hiện tiếng hét / âm thanh méo ---
# Dựa trên RMS energy peak: đoạn nào có peak quá cao so với trung bình
# Hệ số: peak_rms / mean_rms > SCREAM_RATIO → bị loại
SCREAM_RATIO_THRESHOLD = 5.0

# --- Clipping detection (âm thanh bị méo/vỡ) ---
# % mẫu audio vượt ngưỡng biên độ 0.99 → loại
MAX_CLIPPING_PERCENT = 0.8

# --- Silence ratio ---
# Đoạn nào có > X% là im lặng → loại (thường là đoạn chỉ có nhạc)
MAX_SILENCE_RATIO = 0.4  # 40%

# ------------------------------------------------------------------ #
#  BƯỚC 5: CHUẨN HÓA OUTPUT (chuẩn XTTS v2)
# ------------------------------------------------------------------ #
OUTPUT_SAMPLE_RATE = 24000   # Hz — chuẩn XTTS v2
OUTPUT_CHANNELS = 1          # Mono
OUTPUT_FORMAT = "wav"        # WAV — lossless, bắt buộc cho training

# Normalize âm lượng về mức chuẩn (dBFS)
TARGET_LUFS = -20.0

# Thêm padding im lặng ở đầu/cuối mỗi đoạn (ms)
SILENCE_PAD_MS = 50

# ------------------------------------------------------------------ #
#  LOGGING & DEBUG
# ------------------------------------------------------------------ #
LOG_FILE = "./pipeline.log"

# True: lưu file RTTM diarization để kiểm tra bằng tay
SAVE_RTTM = True

# True: lưu audio của TẤT CẢ speaker (không chỉ Ngạn) để debug
SAVE_ALL_SPEAKERS_DEBUG = False