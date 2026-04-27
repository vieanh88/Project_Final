# B3: prepare_ood/step1_clean_phonemize.py.
Đặt vào: TTS_StyleTTS2/fine-tune/data_pipeline/prepare_ood/step1_clean_phonemize.py
Chạy:
bashpython step1_clean_phonemize.py --input "D:/path/to/raw_ood_texts.txt"

# Test nhanh 200 dòng:
python step1_clean_phonemize.py --input raw_ood_texts.txt --max-lines 200
Cần thêm section vào config.yaml (nếu không dùng --input):
yamlprepare_ood:
  input_file: "D:/HUST_Project/Project_Final/data/raw_ood_texts.txt"
  output_file: "OOD_texts_phoneme.txt"
  convert_numbers_to_words: true
  split_long_sentences: true
  split_max_words: 25
  min_words: 3
  max_words: 60
Script này chạy 2 phase:
Phase 1 (Clean & Split): Đọc 50k dòng raw → loại ngoặc kép, dấu ba chấm, ký tự rườm rà → chuyển số thành chữ → nếu câu quá dài (>25 từ) thì tách thông minh tại dấu .?! thành nhiều câu ngắn hơn → lọc bỏ câu quá ngắn (<3 từ) hoặc quá dài (>60 từ). Lưu file tạm ood_cleaned_texts.txt để debug.
Phase 2 (Phonemize): Chạy từng câu đã clean qua vi2IPA_split → lưu OOD_texts_phoneme.txt. Mỗi dòng là 1 chuỗi phoneme thuần (không có wav_path), đúng format mà StyleTTS2 mong đợi ở trường OOD_data.
Tính năng split_long_sentences rất quan trọng cho truyện ma — vì nhiều đoạn văn thô thường là 1 paragraph dài, cần tách ra thành câu 10-25 từ để JAT training hiệu quả.