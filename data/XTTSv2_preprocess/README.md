# Pipeline Xử Lý Audio — Tách Giọng Nguyễn Ngọc Ngạn

## Mục tiêu
Tự động trích xuất các đoạn giọng đọc sạch (dry voice) của Nguyễn Ngọc Ngạn
từ audio truyện ma có nhạc nền, để làm dataset cho **Fine-tuning XTTS v2**.

---

## Cấu trúc file

```
audio_pipeline/
├── config.py            ← Tất cả cài đặt — chỉnh ở đây trước khi chạy
├── pipeline_modules.py  ← Logic xử lý (không cần sửa)
├── run_pipeline.py      ← Script chính để chạy
├── requirements.txt     ← Dependencies
│
├── raw_audio/           ← ĐẶT FILE AUDIO GỐC VÀO ĐÂY
├── workdir/             ← File trung gian (tự tạo)
└── output_ngan_voice/   ← KẾT QUẢ (tự tạo)
    ├── ten_file_goc/
    │   ├── ten_file_goc_seg0001_12.3-18.7s.wav
    │   ├── ten_file_goc_seg0002_20.1-28.5s.wav
    │   └── ...
    └── metadata.csv
```

---

## Cài đặt

### Bước 1: Cài PyTorch (QUAN TRỌNG — cài trước)
```bash
# Kiểm tra CUDA version của máy
nvidia-smi

# CUDA 11.8
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Bước 2: Cài các thư viện còn lại
```bash
pip install -r requirements.txt
```

### Bước 3: Cài FFmpeg (bắt buộc cho Demucs xử lý MP3)
```bash
# Windows (dùng chocolatey)
choco install ffmpeg

# Ubuntu/Debian
sudo apt install ffmpeg

# macOS
brew install ffmpeg
```

### Bước 4: Accept điều khoản pyannote trên HuggingFace
Truy cập và accept terms tại 3 link này (phải đăng nhập HF):
- https://huggingface.co/pyannote/speaker-diarization-3.1
- https://huggingface.co/pyannote/segmentation-3.0
- https://huggingface.co/pyannote/embedding

---

## Cấu hình

Mở `config.py` và chỉnh các thông số sau:

```python
# Bắt buộc
HF_TOKEN = "hf_xxx..."           # HuggingFace token của bạn
INPUT_DIR = "./raw_audio"         # Thư mục chứa audio gốc

# Thông số quan trọng
SPEAKER_SIMILARITY_THRESHOLD = 0.75  # Giảm xuống 0.65 nếu bỏ sót nhiều
MIN_SEGMENT_DURATION = 6.0           # XTTS v2 yêu cầu tối thiểu 6s
```

**Cách đặt HF token an toàn hơn (không hardcode):**
```bash
# Linux/macOS
export HF_TOKEN=hf_your_token_here

# Windows PowerShell
$env:HF_TOKEN="hf_your_token_here"
```

---

## Chạy Pipeline

### Test với 1 file trước (khuyến nghị)
```bash
python run_pipeline.py --file raw_audio/ten_file.mp3
```

### Xử lý tất cả file
```bash
python run_pipeline.py
```

### Resume nếu bị ngắt giữa chừng
```bash
# Bỏ qua bước 1 (Demucs) nếu đã tách xong
python run_pipeline.py --start-step 2

# Bỏ qua bước 1+2 nếu đã diarize xong
python run_pipeline.py --start-step 3
```

### Xem thống kê kết quả
```bash
python run_pipeline.py --stats-only
```

---

## Quy trình xử lý chi tiết

```
File audio gốc (.mp3/.wav)
         │
         ▼ [Bước 1 - Demucs htdemucs_ft]
    vocals.wav (đã tách nhạc nền)
         │
         ▼ [Bước 2 - pyannote diarization]
    Segments: SPEAKER_00, SPEAKER_01, SPEAKER_02...
    → Xác định dominant speaker = giọng Ngạn (chiếm nhiều nhất)
         │
         ▼ [Bước 3 - Speaker Verification]
    → Bootstrap: lấy N segment tốt nhất của dominant speaker
    → Tính reference embedding (ECAPA-TDNN centroid)
    → Verify TOÀN BỘ segment bằng cosine similarity
    → Giữ lại những đoạn similarity >= threshold
         │
         ▼ [Bước 4 - Quality Filter]
    Loại bỏ:
    ✗ Thời lượng < 6s hoặc > 30s
    ✗ SNR < 20dB (còn nhiều nhạc nền)
    ✗ Tiếng hét / âm thanh méo
    ✗ Clipping (biên độ bão hòa)
    ✗ > 40% im lặng
         │
         ▼ [Bước 5 - Normalize & Export]
    output_ngan_voice/
    ├── *.wav (22050Hz, mono, -20 LUFS)
    └── metadata.csv
```

---

## Yêu cầu XTTS v2

| Thông số | Yêu cầu |
|----------|---------|
| Sample rate | 22050 Hz |
| Channels | Mono |
| Format | WAV (PCM 16-bit) |
| Độ dài mỗi đoạn | 6 - 30 giây |
| Tổng dataset | ≥ 30 phút (khuyến nghị 60+ phút) |
| SNR | > 20 dB |

---

## Xử lý sự cố

### Lỗi "CUDA out of memory" khi chạy Demucs
```python
# Trong config.py:
DEMUCS_CPU_ONLY = True   # Chậm hơn nhưng không cần VRAM
```

### Bỏ sót nhiều giọng Ngạn
```python
# Trong config.py, giảm threshold:
SPEAKER_SIMILARITY_THRESHOLD = 0.65
```

### Quá nhiều đoạn bị loại vì "low_snr"
```python
# Demucs có thể chưa tách sạch hoàn toàn, giảm ngưỡng SNR:
MIN_SNR_DB = 15.0
```

### Kiểm tra kết quả diarization bằng mắt
File RTTM được lưu trong `workdir/TEN_FILE/TEN_FILE.rttm`
Mở bằng Audacity hoặc dùng pyannote Notebook để visualize.

---

## Thời gian ước tính (RTX 3050Ti)

| Bước | 1 giờ audio | Ghi chú |
|------|------------|---------|
| Demucs (GPU) | ~15-20 phút | htdemucs_ft |
| Demucs (CPU) | ~60-90 phút | Fallback |
| Diarization | ~8-12 phút | pyannote 3.1 |
| Verification | ~5-8 phút | ECAPA-TDNN |
| Filter + Export | ~3-5 phút | CPU |
| **Tổng (GPU)** | **~35-45 phút** | |