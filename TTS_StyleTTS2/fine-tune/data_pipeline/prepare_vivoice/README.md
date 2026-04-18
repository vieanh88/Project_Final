# Cấu trúc thư mục là:

```
TTS_StyleTTS2/
├── fine-tune/
│   ├── config/                    ← 3 config stage + prepare_vivoice_config.yaml
│   ├── data_pipeline/
│   │   ├── prepare_ngan/
│   │   ├── prepare_ood/
│   │   └── prepare_vicoice/       ← 5 scripts + config.yaml
│   ├── plbert/
│   ├── train_wrapper.py
│   ├── create_mean_style.py
│   ├── nlp_generator.py
│   └── tts_generator.py
└── StyleTTS2/                     ← repo gốc
```

# **.Đây là file **A2: `step2_extract_audio.py`**.

**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vicoice/step2_extract_audio.py`

**Cài thêm dependencies (nếu chưa có):**
```bash
pip install soundfile librosa tqdm
```

**Chạy:**
```bash
cd TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vicoice
python step2_extract_audio.py
```

**Script này làm gì:**

Nó load dataset ViVoice từ HF cache (đã tải ở Bước 1), lặp qua từng sample, giải mã audio bytes thành numpy array (HF `datasets` làm tự động on-the-fly), resample về 24kHz mono 16-bit PCM, normalize peak, rồi lưu thành `.wav` với tên `vivoice_0000000.wav`, `vivoice_0000001.wav`, v.v.

Đồng thời, nó trích xuất cột `text` thô ra file `workdir/raw_texts.txt` (mỗi dòng 1 câu, index tương ứng với file `.wav`) và file `workdir/wav_paths.txt` (mapping path). Hai file này là đầu vào cho Bước 3 (phonemize).

Có cơ chế `skip_existing` — nếu file `.wav` đã tồn tại thì bỏ qua, cho phép resume nếu bị crash giữa chừng. Cuối cùng in thống kê chi tiết và kiểm tra mẫu 3 file `.wav` đầu tiên.

# **A3: `step3_phonemize.py`**.

**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vicoice/step3_phonemize.py`

**Dependencies (đã có từ step08 của bạn):**
```bash
pip install vinorm viphoneme tqdm
```

**Chạy:**
```bash
python step3_phonemize.py
# Hoặc test nhanh với 100 dòng đầu:
python step3_phonemize.py --max-lines 100
```

**Thiết kế quan trọng:**

Script này copy nguyên khối logic monkey-patch vinorm đã chạy hoàn hảo từ `step08_phonemize.py` của bạn — cùng hàm `_mock_tts_norm`, cùng thứ tự patch `vinorm → viphoneme → vi2IPA_split`. Không gọi subprocess tới step08 mà nhúng trực tiếp logic G2P để tránh overhead và dễ debug.

Điểm khác biệt so với step08 là cơ chế giữ index 1:1: nếu dòng nào G2P thất bại, script ghi `[FAILED]` vào đúng vị trí đó thay vì bỏ qua. Nhờ vậy, dòng thứ N trong `phoneme_texts.txt` luôn tương ứng với dòng thứ N trong `raw_texts.txt` và file `.wav` thứ N. Bước 5 (make_filelist) sẽ lọc bỏ các dòng `[FAILED]` khi ghép.

Ngoài ra có thêm hàm `clean_text()` để loại ngoặc kép, ngoặc đơn, chuẩn hóa dấu câu liên tiếp trước khi đưa vào viphoneme. File `phonemize_errors.txt` ghi chi tiết từng dòng lỗi để debug.

# **A4: `step4_build_vocab.py`**.

**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vicoice/step4_build_vocab.py`

**Không cần cài thêm dependencies** — chỉ dùng standard library (`json`, `collections.Counter`).

**Chạy:**
```bash
python step4_build_vocab.py

# Hoặc kèm thêm file phoneme từ prepare_ngan, prepare_ood (nếu đã có):
python step4_build_vocab.py --extra-phoneme-files ../prepare_ngan/ngan_phoneme.txt ../prepare_ood/OOD_texts_phoneme.txt
```

**Thiết kế quan trọng:**

File `phoneme_vocab.json` output sẽ có cấu trúc:
```json
{
  "_metadata": { "n_token": 185, "num_special_tokens": 4, ... },
  "n_token": 185,
  "char_to_id": { "<pad>": 0, "<unk>": 1, " ": 4, "a": 5, ... },
  "id_to_char": { "0": "<pad>", "1": "<unk>", ... },
  "char_frequencies": { " ": 2500000, "a": 1800000, ... }
}
```

Giá trị `n_token` ở đây chính là con số mà `train_wrapper.py` sẽ đọc và inject vào `model_params.n_token` trong 3 file config stage, thay thế giá trị 178 mặc định của repo gốc (vì tiếng Anh và tiếng Việt có bộ phoneme khác nhau).

4 special tokens (`<pad>`, `<unk>`, `<bos>`, `<eos>`) được đặt ở ID 0-3. Các ký tự IPA thực tế sắp xếp theo tần suất giảm dần — ký tự phổ biến nhất có ID nhỏ nhất, giúp embedding lookup hiệu quả hơn.

Script hỗ trợ `--extra-phoneme-files` để sau này khi đã chạy xong prepare_ngan và prepare_ood, bạn có thể rebuild vocab đầy đủ nhất bằng cách truyền thêm các file phoneme đó vào. Điều này đảm bảo vocab cover 100% ký tự IPA từ cả 3 nguồn dữ liệu.

Cuối log sẽ in top 30 ký tự phổ biến nhất và 10 ký tự hiếm nhất để bạn kiểm tra có noise hay ký tự lạ lọt vào không.

# **A5: `step5_make_filelist.py`** — file cuối cùng của pipeline `prepare_vivoice`.

**Vị trí:** `TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vicoice/step5_make_filelist.py`

**Chạy:**
```bash
python step5_make_filelist.py

# Hoặc bỏ qua kiểm tra wav tồn tại (nhanh hơn nếu chắc chắn data đúng):
python step5_make_filelist.py --no-verify-wav
```

**Script này làm gì:**

Đọc 2 file trung gian `wav_paths.txt` + `phoneme_texts.txt` (cả hai có cùng số dòng, index 1:1), ghép thành format chuẩn StyleTTS2: `wav_path|phoneme|speaker_id`. Sau đó lọc bỏ các dòng không hợp lệ (phoneme `[FAILED]`, rỗng, quá ngắn/dài, wav không tồn tại), shuffle với seed cố định, rồi split theo tỷ lệ 95/5 thành `vivoice_train_list.txt` và `vivoice_val_list.txt`.

**Các bộ lọc chất lượng:**
- `[FAILED]` từ step3 → loại bỏ
- Phoneme < 3 ký tự → loại (nhiễu/noise)
- Phoneme > 5000 ký tự → loại (bất thường)
- File `.wav` không tồn tại → loại (flag `--no-verify-wav` để tắt)

**Speaker ID:** ViVoice dùng `speaker_id = 1`, để dành `0` cho Bác Ngạn — đúng theo thiết kế trong tài liệu.

Cuối log in thống kê phân phối độ dài phoneme (min/max/mean/median/stdev) để bạn đánh giá chất lượng data.

---

**Toàn bộ pipeline `prepare_vivoice` (A0-A5) đã hoàn tất!** Tóm tắt thứ tự chạy:

```
step1_download.py       → Tải parquet từ HF
step2_extract_audio.py  → Decode → 24kHz mono .wav + raw_texts.txt
step3_phonemize.py      → Text → IPA phoneme
step4_build_vocab.py    → Quét phoneme → phoneme_vocab.json (n_token)
step5_make_filelist.py  → Ghép wav|phoneme|speaker_id → train/val list
```
