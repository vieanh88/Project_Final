# FILE A1: step1_rephonemize_lite.py
**Mục tiêu:** Đọc text gốc tiếng Việt của Ngạn từ pipeline cũ → phonemize bằng viphoneme.vi2IPA_split → replace _ thành space (vì vocab của StyleTTS2-lite-vi không có ký tự _) → ghi output vào TTS_StyleTTS2-lite-vi/output/.

**Logic chính:**
1. Reuse pattern khắc phục lỗi vinorm/charmap từ step08_phonemize.py cũ của bạn (giữ nguyên monkey-patch để tránh WinError 193).
2. Đọc 2 file filelist_train.txt và filelist_val.txt.
3. Mỗi dòng wav_path|text → phonemize → replace _ → space → output wav_path|phoneme.
4. Quan trọng: KHÔNG normalize backslash ở bước này (bước A2 sẽ làm — vì A2 mới biết root_path để chuyển đổi đường dẫn tương đối).
5. Skip dòng lỗi G2P, log đầy đủ.

**Path đầu vào/đầu ra (theo cấu trúc bạn confirm):**

- Input: `data/StyleTTS2_preprocess/output_dataset/filelist_{train,val}.txt`
- Output: `TTS_StyleTTS2-lite-vi/output/ngan_{train,val}_phoneme_raw.txt`

(Tôi đặt suffix _phoneme_raw để A2 sẽ tạo file final ngan_{train,val}_lite.txt — phân biệt rõ output trung gian vs output cuối cùng.)

# Tóm tắt file A1

**Điểm thiết kế chính:**

1. Path resolution thông minh: Dùng __file__ để self-locate, tự suy ra `PROJECT_ROOT` và `PROJECT_FINAL_ROOT` từ vị trí script. Không hardcode đường dẫn `D:/HUST_Project/...` → file sẽ chạy được trên bất kỳ máy nào miễn cấu trúc thư mục đúng.
2. Tận dụng tối đa pipeline cũ: Giữ nguyên monkey-patch vinorm/viphoneme đã verify hoạt động trên Windows trong `step08_phonemize.py`.
3. Chỉ làm 1 việc duy nhất: Phonemize + replace `_ → space`. KHÔNG động vào path normalization, KHÔNG validate vocab, KHÔNG split. Mọi việc đó đẩy sang A2 — separation of concerns.
4. Output là intermediate file `(*_phoneme_raw.txt)`: có cùng format `wav_path|phoneme` với input nhưng phoneme đã sạch. wav_path GIỮ NGUYÊN backslash Windows để A2 quyết định cách normalize.
5. Self-test với câu mẫu trước khi xử lý hàng loạt — fail fast nếu vinorm bị lỗi cài đặt.
6. Log có "[SAMPLE]" in ra ví dụ đầu tiên thành công để bạn kiểm tra trực quan format output có đúng không.


# Cách test file A1
**Bước 1: Tạo cấu trúc thư mục mới:**
```
HUST_Project/Project_Final/TTS_StyleTTS2-lite-vi/
└── data_pipeline/
    └── prepare_ngan_lite/
        └── step1_rephonemize_lite.py    (copy file này vào đây)
```
**Bước 2: Đảm bảo data Ngạn đã ở đúng vị trí:**
```
HUST_Project/Project_Final/data/StyleTTS2_preprocess/output_dataset/
├── filelist_train.txt
├── filelist_val.txt
└── wavs
```
**Bước 3:** Activate environment có sẵn `viphoneme + vinorm` (cùng env bạn đã chạy `step08_phonemize.py`).

**Bước 4:** Chạy:
```bash
cd D:\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi
python -X utf8 data_pipeline\prepare_ngan_lite\step1_rephonemize_lite.py
```
**Bước 5: Kết quả mong đợi:**

- 2 file mới sinh ra trong TTS_StyleTTS2-lite-vi/output/:
    - ngan_train_phoneme_raw.txt
    - ngan_val_phoneme_raw.txt

- File log: `TTS_StyleTTS2-lite-vi/logs/step1_rephonemize_lite.log`
- Log có dòng [SAMPLE] cho thấy 1 ví dụ phonemize thành công.

Format dòng output mong đợi (so với input gốc):
```
# Input:
output_dataset\wavs\ngan_00070.wav|Vừa nhà khói, vừa dơ bao thuốc ra trước mặt...

# Output:
output_dataset\wavs\ngan_00070.wav|v ɤ 2 ɲ a 2 x ɔ j 4 , v ɤ 2 z ɤ 1 b a w 1 ...
```
(không còn dấu `_` trong chuỗi phoneme, các phoneme cách nhau bằng space)

# FILE A2: step2_make_filelist_lite.py
**Mục tiêu:** Đọc 2 file _phoneme_raw.txt từ A1 → validate vocab character-level → normalize path → filter audio quá ngắn → ghi filelist final.

**Logic chi tiết:**

Build symbol_dict từ config.yaml giống hệt cách inference.py làm (5 nguồn: pad, punctuation, letters, letters_ipa, extend).
Validate từng ký tự trong chuỗi phoneme. Nếu có ký tự lạ → log + skip dòng (vì TextCleaner sẽ silently drop, gây sai alignment).
Normalize path: backslash \ → forward slash / (cho Linux/Kaggle compatibility).
Filter audio < 0.5s: dùng soundfile.info() để check duration (BatchSampler bỏ audio < 20 frames mel = 20 × 300/24000 ≈ 0.25s, nên cut ở 0.5s để safety margin).
Path resolution: Convert wav_path từ format pipeline cũ (output_dataset\wavs\ngan_xxx.wav) → relative tới ROOT_PATH mà Kaggle/training sẽ set.

Quan trọng về root_path: Trong meldataset.py, line wave, sr = sf.read(osp.join(self.root_path, wave_path)). Nên trong filelist, wave_path phải là đường dẫn tương đối tới root_path.

Trên Kaggle, root_path sẽ là /kaggle/input/<dataset-name>/
Wav nằm ở /kaggle/input/<dataset-name>/wavs/ngan_00001.wav
Vậy filelist nên ghi wavs/ngan_00001.wav|<phoneme> (không có prefix output_dataset/)

→ A2 cần strip prefix output_dataset/ khỏi path để có path bắt đầu từ wavs/.
Audio check: A2 cần đọc file audio thực tế (từ data/StyleTTS2_preprocess/output_dataset/wavs/) để check duration. Tôi sẽ làm option có thể bật/tắt vì với 200 phút data có thể tốn thời gian.Created a file, read a fileCreated a file, read a file
Tóm tắt file A2
Điểm thiết kế chính:

Build vocab CHÍNH XÁC như inference.py của lite-vi (đã verify qua source code thực tế trên HF):

python   symbols = pad + punctuation + letters + letters_ipa + extend
   symbol_dict = {ch: idx for idx, ch in enumerate(symbols)}
   n_token = len(symbol_dict) + 1   # = 189 cho lite-vi
Không hardcode, đọc động từ config.yaml của model bạn download → đảm bảo 100% khớp model thực tế.

Validate character-level: Vì meldataset.TextCleaner iterate từng char và silently skip ký tự lạ (KeyError → continue), nếu không validate trước, dữ liệu sai sẽ vào model mà không có warning. A2 catch toàn bộ ký tự lạ + thống kê tần suất → bạn biết ngay nếu pipeline cũ leak ký tự gì.
Path normalization 3 bước: backslash → forward slash, strip prefix output_dataset/, strip leading slash. Output ví dụ:

Input  : output_dataset\wavs\ngan_00070.wav
Output : wavs/ngan_00070.wav
Khi training trên Kaggle, set root_path: /kaggle/input/<dataset>/ → ghép thành /kaggle/input/<dataset>/wavs/ngan_00070.wav ✓


Audio duration check với 3 candidate paths để robust với mọi cách user organize wav files. Có flag --no-audio-check để chạy nhanh nếu bạn không muốn đọc file (hữu ích lúc test/iterate).
Cảnh báo khi n_token != 189: Nếu user vô tình dùng config sai (ví dụ config gốc 178 tokens của LibriTTS) → sẽ báo warning ngay.
Log thống kê chi tiết: ok, vocab_fail, audio_short, audio_missing, format_bad — tách bạch để debug nhanh.


Cách test file A2
Bước 1: Download config.yaml của lite-vi:

Truy cập: https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/blob/main/Models/config.yaml
Bấm "download raw file" → lưu về TTS_StyleTTS2-lite-vi/configs/config.yaml

Bước 2: Đặt file A2 vào đúng vị trí:
TTS_StyleTTS2-lite-vi/data_pipeline/prepare_ngan_lite/step2_make_filelist_lite.py
Bước 3: Đảm bảo A1 đã chạy xong và có:
TTS_StyleTTS2-lite-vi/output/ngan_train_phoneme_raw.txt
TTS_StyleTTS2-lite-vi/output/ngan_val_phoneme_raw.txt
Bước 4: Chạy:
bashcd D:\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi
python -X utf8 data_pipeline\prepare_ngan_lite\step2_make_filelist_lite.py
Hoặc nếu muốn nhanh (skip audio duration check):
bashpython -X utf8 data_pipeline\prepare_ngan_lite\step2_make_filelist_lite.py --no-audio-check
Bước 5: Kết quả mong đợi:

2 file mới sinh ra trong TTS_StyleTTS2-lite-vi/output/:

ngan_train_lite.txt (FILE FINAL)
ngan_val_lite.txt (FILE FINAL)


Format mỗi dòng: wavs/ngan_00070.wav|v ɤ 2 ɲ a 2 x ɔ j 4 , ...
Log có dòng [SAMPLE] cho thấy normalize đúng.
Built symbol_dict: 189 unique symbols, n_token = 189
Tổng OK: ~5000-6000 dòng (ước lượng cho 200 phút data, mỗi sample ~2-4s).