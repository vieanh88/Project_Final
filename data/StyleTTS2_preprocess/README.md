# Bước 1: Một số điểm đáng chú ý:
config.yaml — File duy nhất cho tất cả 7 bước, có section riêng cho từng bước. Bạn chỉ cần sửa ở đây, không cần đụng vào code.
step01_vocal_isolation.py — 3 tính năng quan trọng so với pipeline cũ:

Cache thông minh: Tự phát hiện file đã tách rồi → skip, không chạy lại Demucs tốn thời gian
Flatten output: Demucs tạo cấu trúc thư mục lồng nhau _demucs_raw/htdemucs_ft/ten_file/vocals.wav → script tự copy ra thành step01_vocals/ten_file_vocals.wav để bước 2 dùng thẳng
--dry-run: Xem trước file nào cần xử lý mà không tốn thời gian

Cách chạy:
```bash
# Test 1 file trước
python step01_vocal_isolation.py --input raw_audio/ten_file.mp3

# Xử lý tất cả
python step01_vocal_isolation.py
```
# Bước 2: Một số điểm kỹ thuật đáng chú ý:
Tại sao bước 2 quan trọng với trường hợp của bạn:
Log của bạn cho thấy low_snr(9.6dB < 10.0dB) — những đoạn này không phải là Demucs tách kém, mà là reverb và room echo còn sót lại sau Demucs. DeepFilterNet3 được thiết kế chính xác để xử lý loại noise này, nên sau bước 2 điểm SNR của các đoạn đó sẽ tăng lên đáng kể.
Chunk processing (60s/chunk): File audio truyện Ngạn thường rất dài (40-60 phút). Script tự chia nhỏ để tránh OOM, sau đó ghép lại seamlessly.
attenuation_limit: 0.97 trong config — đây là mức lọc mạnh (97dB). Nếu bạn nghe thấy giọng bị "robotic" hoặc mất âm sắc tự nhiên sau bước này, hãy giảm xuống 0.85.

# Bước 5: Một số thiết kế quan trọng:
3 lớp lọc transcript — tránh data poisoning
Whisper đôi khi "hallucinate" — tạo ra văn bản không khớp với audio, đặc biệt với đoạn im lặng hoặc noise. Script có 3 lớp lọc:
LớpĐiều kiện loạiLý do1Dưới min_words từQuá ngắn, không đủ ngữ cảnh2no_speech_prob > 0.8Whisper tự đánh giá đây là silence3avg_logprob < -1.5Confidence thấp — kết quả không đáng tin
condition_on_previous_text=False
Quan trọng với dataset dạng slice rời. Nếu để True, Whisper dùng context từ segment trước → dễ hallucinate câu tiếp theo khi không có audio thực sự.
--resume flag rất hữu ích
Với 200+ slice, nếu bị ngắt điện giữa chừng:
```bash
bashpython step05_transcription.py --resume
# Tự bỏ qua các file đã có .txt, chỉ transcribe phần còn thiếu
```
Output format LJSpeech
File filelist.txt xuất ra dạng:
```bash
workdir/step05_transcribed/ten_file/ten_file_0001.wav|Đây là nội dung câu nói...
workdir/step05_transcribed/ten_file/ten_file_0002.wav|Bóng đen xuất hiện phía sau...
```
Đây là format chuẩn được StyleTTS2, VITS, và hầu hết TTS framework đọc trực tiếp.

# Bước 6: Một số điểm thiết kế cần lưu ý:

DNSMOS chạy hoàn toàn offline sau lần đầu
Script tự download 2 file .onnx (~10MB) từ Microsoft DNS-Challenge repo vào thư mục dnsmos_models/. Sau lần đầu không cần internet nữa. Nếu mạng bị chặn, bạn có thể download thủ công và đặt vào dnsmos_models/:

- sig_bak_ovr.onnx
- model_v8.onnx

3 output file hữu ích

quality_scores.csv: Tất cả slices + MOS score (kể cả fail) — dùng để phân tích

quality_manifest.csv: Chỉ slice pass + đầy đủ metadata

filelist.txt: LJSpeech format — input trực tiếp cho StyleTTS2
H
istogram MOS trong log

Mỗi lần chạy sẽ in phân phối score dạng:

  3.0-3.5: ████ 12 (8.5%)

  3.5-4.0: ████████████████ 48 (34.0%)  ← ngưỡng

  4.0-4.5: ██████████████████████ 65 (46.1%)

Giúp bạn thấy ngay nên điều chỉnh ngưỡng lên hay xuống mà không cần mở CSV.

Flag --score-only tiện khi calibrate
```bash
# Xem điểm tất cả slices mà không copy file (nhanh hơn)
python step06_quality_check.py --score-only
# Xem quality_scores.csv → quyết định threshold
# Sau đó chạy lại không có --score-only để copy file thực sự
```

# Bước 7: Tóm tắt:
Điểm thiết kế quan trọng của bước 7

Pyloudnorm → RMS fallback: Script thử dùng pyloudnorm (chuẩn EBU R128 thực sự) trước. Nếu chưa cài thì tự fallback về RMS normalization — không bao giờ crash vì thiếu thư viện optional.

Trim trước, pad sau: librosa.effects.trim cắt bỏ silence thừa ở đầu/cuối (do bước 3 padding để lại), sau đó mới thêm 50ms padding sạch. Tránh tình trạng file có 200ms silence đầu rồi thêm 50ms nữa thành 250ms.

Train/val split với seed cố định: random.seed(42) đảm bảo mỗi lần chạy lại ra cùng split — reproducible khi báo cáo kết quả.

dataset_info.json in kết quả sẵn sàng:
```json
json"styletts2_readiness": {
  "ready": true,
  "note": "✅ Đủ điều kiện fine-tuning"
}
```

Cấu trúc output_dataset/ cuối cùng
```bash
output_dataset/
├── wavs/
│   ├── ngan_00001.wav   ← 24kHz / mono / 16-bit / -20 LUFS
│   ├── ngan_00002.wav
│   └── ...
├── filelist_train.txt   ← StyleTTS2 đọc trực tiếp
├── filelist_val.txt
├── filelist_all.txt
└── dataset_info.json
```
Thứ tự chạy pipeline
```bash
python step01_vocal_isolation.py
python step02_audio_restoration.py
python step03_vad_slicing.py
python step04_speaker_filter.py
python step05_transcription.py
python step06_quality_check.py
python step07_formatting.py
```
Nếu gặp lỗi ở bước nào, sửa config rồi chạy lại đúng bước đó — các bước khác không bị ảnh hưởng nhờ cơ chế cache và manifest độc lập.