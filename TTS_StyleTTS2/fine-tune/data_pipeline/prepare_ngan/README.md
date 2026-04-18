# **B1: `prepare_ngan/step1_phonemize.py`**.

**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_ngan/step1_phonemize.py`

**Chạy:**
```bash
# Cách 1: Chỉ định trực tiếp đường dẫn output_dataset của pipeline 7 bước
python step1_phonemize.py --ngan-dir "D:/HUST_Project/Project_Final/data/StyleTTS2_preprocess/output_dataset"

# Cách 2: Đặt trong config.yaml (xem bên dưới)
python step1_phonemize.py --config config.yaml

# Test nhanh 50 dòng:
python step1_phonemize.py --ngan-dir "path/to/output_dataset" --max-lines 50
```

**Cần thêm section sau vào config.yaml** (file config chung hoặc tạo config riêng cho prepare_ngan):
```yaml
prepare_ngan:
  dataset_dir: "D:/HUST_Project/Project_Final/data/StyleTTS2_preprocess/output_dataset"
  input_filelists:
    - "filelist_train.txt"
    - "filelist_val.txt"
  convert_numbers_to_words: true
  skip_existing: true

paths:
  work_dir: "./workdir"
```

**Điểm khác biệt so với step3 (ViVoice):**

Script này có thêm hàm `clean_ngan_text()` làm sạch mạnh hơn — đặc biệt cho transcript Whisper của audiobook, nơi thường có dấu ba chấm `...`, ngoặc kép lồng nhau, gạch ngang em-dash `—`, và các ký tự đặc biệt khác. Quan trọng nhất là hàm `digits_to_words()` chuyển số thành chữ tiếng Việt (ví dụ: "phòng 304" → "phòng ba không bốn") vì viphoneme không xử lý tốt chữ số. Theo tài liệu thiết kế, đây là yêu cầu bắt buộc: *"Xóa các ký tự đặc biệt, số (nếu còn sót) trước khi đưa vào viphoneme."*

Output là `ngan_train_phoneme.txt` và `ngan_val_phoneme.txt` với format `wav_path|phoneme` — bước tiếp theo B2 sẽ append `|0` (speaker_id Bác Ngạn) vào.

# **B2: `prepare_ngan/step2_make_filelist.py`**.
**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_ngan/step2_make_filelist.py`

**Chạy:**
```bash
python step2_make_filelist.py
python step2_make_filelist.py --ngan-dir "D:/path/to/output_dataset"
python step2_make_filelist.py --no-verify-wav   # bỏ qua check wav tồn tại
```

**Script này làm gì:**

Đọc 2 file phoneme từ Bước 1 (`ngan_train_phoneme.txt`, `ngan_val_phoneme.txt`), validate từng record (loại phoneme rỗng, quá ngắn, quá dài, wav không tồn tại), rồi append `|0` (speaker_id Bác Ngạn) vào cuối mỗi dòng. Output là `ngan_train_list.txt` và `ngan_val_list.txt` với format chuẩn StyleTTS2: `wav_path|phoneme|0`.

**Tính năng kiểm tra chéo vocab (OOV detection):**

Nếu bạn trỏ `vocab_file` tới `phoneme_vocab.json` đã tạo từ ViVoice, script sẽ quét tất cả phoneme của Bác Ngạn và phát hiện ký tự IPA nào chưa có trong vocab. Nếu có OOV, nó sẽ cảnh báo bạn cần rebuild vocab bằng:
```bash
# Chạy lại step4 với thêm file phoneme Ngạn
python step4_build_vocab.py --extra-phoneme-files ../prepare_ngan/workdir/ngan_train_phoneme.txt
```

Nếu input chỉ có 1 file duy nhất (không tách sẵn train/val), script sẽ tự động shuffle + split 95/5. Nếu đã tách sẵn thì giữ nguyên.