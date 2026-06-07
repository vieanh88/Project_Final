# Tóm tắt file D0

## Điểm thiết kế chính

1. **1 class `StyleTTS2LiteVNInference`** với 4 method public + đầy đủ docstring:
   - `__init__` — load 1 lần (model + config + vocab + helpers)
   - `compute_style(audio_path)` → cache được, dùng cho narrator/character_male/character_female
   - `text_to_phoneme(text)` → tách riêng để D2 (Gemini) có thể preview phoneme
   - `synthesize(text_or_phoneme, style, ...)` → wav numpy 24kHz

2. **FP16 autocast cho RTX 3050Ti 4GB**: giảm VRAM ~50%, **nhưng** duration prediction và outlier replacement vẫn ở FP32 (nhạy với numerical precision). Có flag `--no-fp16` nếu cần debug.

3. **Auto CPU fallback** khi CUDA OOM: tạm move model sang CPU → synthesize → restore device. Chậm hơn ~10-20x nhưng KHÔNG crash app khi câu dài. Đặc biệt quan trọng cho UI demo — không muốn crash giữa demo.

4. **Strip `module.` prefix** khi load checkpoint — vì train.py wrap modules bằng `MyDataParallel` → state_dict keys có prefix `module.`. Code handle cả 2 trường hợp (strict=True trước, retry strict=False nếu fail).

5. **Filter chỉ 4 modules cần** (decoder, predictor, text_encoder, style_encoder) — checkpoint có 8 modules nhưng inference chỉ cần 4.

6. **Phoneme normalize logic giống hệt step1b** đã verified: strip `\u032A` (combining dental bridge), thay hyphens (`-`, `\u2010-\u2014`, `\u2212`) bằng space.

7. **Smoke test CLI** ngay trong file để test độc lập trước khi build D1-D3 và UI.

8. **Tách `already_phonemized`**: D3 sẽ phonemize 1 lần cho toàn bộ script.json, sau đó loop synthesize → khỏi phải phonemize lại. Tiết kiệm thời gian.

---

# Cách test D0

## Bước 1: Đảm bảo có repo lite-vi + config + audio reference

Bạn cần 3 thứ:

1. **Folder repo gốc** (clone từ GitHub, có `models.py`, `Modules/`, `meldataset.py`):
   ```bash
   cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi
   git clone https://github.com/dangtr0408/StyleTTS2-lite.git
   ```
   
   → Folder `StyleTTS2-lite/` xuất hiện.

   **QUAN TRỌNG**: Apply patch JDC `.squeeze()` → `.squeeze(-1)` giống Kaggle (vì class JDCNet được khởi tạo trong `build_model`, sẽ load weights nhưng KHÔNG được dùng cho inference — tuy nhiên import vẫn chạy code class). Hoặc đơn giản: copy folder `StyleTTS2-lite/` từ Kaggle về (đã có patch).

2. **Config file**: dùng `config.yaml` gốc lite-vi (tải từ HF) HOẶC `config_ngan_kaggle.yml` (đã generate ở Kaggle). Cả 2 có cùng `symbol` section → tương đương. Tải về:
   ```bash
   curl -L https://huggingface.co/dangtr0408/StyleTTS2-lite-vi/resolve/main/Models/config.yaml -o D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\Models\config.yaml
   ```

3. **Audio reference**: copy 1 file `.wav` của Ngạn từ dataset đã clean:
   ```
   D:\Documents\HUST\HUST_Project\Project_Final\data\StyleTTS2_preprocess\output_dataset\wavs\ngan_00100.wav
   ```
   
   (Chọn file ngắn ~5-10s, giọng rõ, không có background music.)

## Bước 2: Cài deps đầy đủ trên local

Mở terminal trong thư mục `TTS_StyleTTS2-lite-vi/`:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install munch noisereduce phonemizer espeakng-loader nltk librosa soundfile pyyaml
```

## Bước 3: Đặt file D0 và chạy smoke test

Copy file `inference_engine.py` vào:
```
D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\inference_engine.py
```

Chạy smoke test:
```bash
cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi

python inference_engine.py ^
  --checkpoint Models\epoch_00030.pth ^
  --repo StyleTTS2-lite ^
  --config Models\config.yaml ^
  --ref data\..\data\StyleTTS2_preprocess\output_dataset\wavs\ngan_00100.wav ^
  --text "Đêm hôm ấy, trời tối đen như mực, không một tiếng động." ^
  --out smoke_test.wav
```

**(Chỉnh path `--ref` cho đúng vị trí thực tế của bạn — tôi đoán đường dẫn data có thể khác.)**

## Bước 4: Mong đợi output

```
[INFO] Device       : cuda
[INFO] FP16 autocast: True
[INFO] Vocab        : 189 symbols, n_token=189
[INFO] Loading checkpoint: ...\epoch_00030.pth
[INFO]   Size: 1600.0 MB
[INFO]   Trained: epoch=30, iters=XXXXX, val_loss=0.32...
[INFO]   decoder       : 54.29 M params
[INFO]   predictor     : 16.19 M params
[INFO]   text_encoder  : 5.61 M params
[INFO]   style_encoder : 13.85 M params
[INFO]   TOTAL         : 89.94 M params
[INFO] ✅ Inference engine ready.

Engine info:
  device              : cuda
  fp16                : True
  n_token             : 189
  ...

Text  : Đêm hôm ấy, trời tối đen như mực, không một tiếng động.
Phn   : ɗem˧ hom˧ ɐɪ˨ ...

Compute style từ: ...\ngan_00100.wav
  Style shape: (1, 128)

Synthesize...
  Wav shape : (XXXXX,)  (3.45s audio)
  Time      : 0.5s
  RTF       : 0.15x  (lower = faster than realtime)

✅ Saved: smoke_test.wav
```

## Bước 5: Mở `smoke_test.wav` nghe

**Kỳ vọng**: Giọng giống Ngạn (trầm, chậm rãi, có sắc thái kể chuyện). Câu rõ ràng, đúng tiếng Việt.

**Báo lại cho tôi** (5-7 dòng đủ):
- ✅ Smoke test pass: engine info, RTF, kích thước wav, **chất lượng audio** (nghe có giống Ngạn không?)
- ❌ Lỗi: traceback đầy đủ + stage nào (load checkpoint? compute_style? synthesize?)


## Cân nhắc thiết kế D1

Bạn chọn **option (ii) — tự cung cấp file `.wav` giọng nữ**. Vậy D1 không phải là "download from ViVoice", mà là **script để VERIFY file `.wav` giọng nữ bạn cung cấp** — quan trọng vì:

1. Audio reference phải đạt chuẩn để `compute_style()` cho ra style vector hợp lệ.
2. Nếu file nữ chất lượng kém → style méo → `character_female` đọc sẽ "giả nữ" không tự nhiên.
3. Phải test SYNTHESIZE thật với 1 câu để xác nhận, không chỉ check duration/SR.

### Yêu cầu cho file nữ reference

Dựa trên hành vi của `compute_style()` (đã đọc kỹ code D0):
- Phải decode được (wav/mp3/flac)
- Sau `librosa.effects.trim(top_db=30)`, phải ≥ 0.5s (engine sẽ raise nếu không)
- Lý tưởng: 5-15s, giọng rõ, 1 speaker, ít noise/nhạc nền
- Sample rate bất kỳ (engine tự resample về 24kHz)
- Mono hoặc stereo (engine tự convert mono)

### 3 việc D1 sẽ làm

1. **Health check file**: kiểm tra format, duration, peak amplitude, có chứa speech không (RMS check)
2. **Test compute_style**: extract style → verify shape `(1, 128)` + có giá trị numerical valid (không NaN/Inf)
3. **Synthesize 3 câu test**: dùng style từ file nữ + 3 câu tiếng Việt đa dạng (kể chuyện, hỏi đáp, cảm thán) → render ra 3 wav để bạn nghe đánh giá

### Decision quan trọng: chuẩn hóa giọng nữ trước khi dùng

Bạn có thể có 1 trong 2 loại file:
- **(a)** File "sạch sẵn" — đã clean, 24kHz, 1 speaker
- **(b)** File "thô" — từ YouTube, có nhạc nền, multi-speaker, sample rate khác

D1 sẽ **phát hiện** và đề xuất bạn xử lý lại nếu là loại (b). KHÔNG tự động clean (Demucs/DeepFilterNet quá nặng cho 1 script utility) — chỉ warn + cho biết cách xử lý.

### CLI design

```bash
python inference/download_female_ref.py --female-ref female_voice.wav
```

Có 1 flag optional `--male-ref` để cùng lúc test giọng Ngạn (verify D0 vẫn work):

```bash
python inference/download_female_ref.py --female-ref female_voice.wav --male-ref ngan_00002.wav
```

------

## Tóm tắt file D1

### Điểm thiết kế chính

1. **3 stage testing — fail fast**:
   - Stage 1: Health check (không cần model) → fail sớm nếu file bad
   - Stage 2: `compute_style()` → check NaN/Inf trong tensor
   - Stage 3: Render 3 câu test với 3 ngữ điệu khác nhau

2. **3 câu test có mục đích**:
   - "Kể chuyện" — trung tính, baseline ngữ điệu
   - "Lời thoại nữ (hỏi)" — câu hỏi → test prosody upward intonation
   - "Cảm thán (sợ hãi)" — trải dài cảm xúc → test sustained emotion (rất quan trọng cho domain horror)

3. **Health check kiểm tra 8 thứ quan trọng**:
   - File tồn tại + decode được
   - Sample rate, channels, duration, format
   - RMS energy (phát hiện silence/almost silence)
   - Peak (phát hiện clipping)
   - Duration sau `librosa.effects.trim(top_db=30)` — đây là duration THỰC tế engine sẽ dùng
   - Warnings cho từng cases edge (sample rate thấp, mono/stereo, quá ngắn/dài)

4. **Tách warnings vs errors**:
   - Warnings: vẫn dùng được, nhưng review
   - Errors: dừng ngay, không tiếp tục stage tiếp theo
   - Gợi ý fix cho từng loại lỗi (vd: ffmpeg command)

5. **Comparison mode optional**: `--male-ref` để render cùng 3 câu nhưng với reference Ngạn → bạn nghe **side-by-side** xem giọng nữ có khác biệt rõ với giọng Ngạn không (nếu 2 giọng nghe giống nhau → có vấn đề với style encoder).

6. **`--skip-synthesize` flag**: chỉ chạy health check (5 giây) — hữu ích khi bạn muốn test nhanh 1 loạt file `.wav` ứng viên trước khi quyết định dùng cái nào.

7. **Output có cấu trúc**: tất cả wav vào `output/female_ref_test/` với tên rõ ràng:
   ```
   female_narration.wav, female_dialogue.wav, female_exclamation.wav
   male_ngan_narration.wav, male_ngan_dialogue.wav, male_ngan_exclamation.wav
   ```

### Lý do thiết kế thận trọng

Trong inference pipeline cuối, **chất lượng giọng character_female phụ thuộc HOÀN TOÀN vào 1 file reference duy nhất** (style encoder reference-based). Một file kém → toàn bộ audiobook horror sẽ bị bad ở character_female. D1 này là barrier kiểm soát chất lượng trước khi qua D2/D3.

---

## Cách test D1

### Bước 1: Chuẩn bị file giọng nữ

Bạn cần 1 file `.wav` giọng nữ tiếng Việt, lý tưởng:
- 5-15 giây
- 1 speaker, giọng rõ
- Không nhạc nền, không noise
- Sample rate ≥ 22050 Hz (tốt nhất 24000)

**Gợi ý nguồn**:
- YouTube: tìm "đọc truyện ma giọng nữ" → tải đoạn ngắn → cắt 5-15s
- Tự record qua microphone laptop
- Lấy 1 sample từ dataset ViVoice (giọng nữ kể chuyện)

**Convert sang format chuẩn** (nếu cần):
```bash
ffmpeg -i input.mp3 -ar 24000 -ac 1 -sample_fmt s16 female_ref.wav
```

Đặt file ở vị trí dễ nhớ, ví dụ: `D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\Models\references\female_ref.wav`

### Bước 2: Copy D1 vào đúng folder

```
TTS_StyleTTS2-lite-vi/inference/download_female_ref.py
```

(Cùng folder với `inference_engine.py`.)

### Bước 3: Chạy chỉ health check trước (5 giây)

```bash
cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi

python inference/download_female_ref.py ^
  --female-ref Models\references\female_ref.wav ^
  --skip-synthesize
```

**Mong đợi**: Health check passes hoặc warnings nhẹ. Nếu errors → fix file rồi chạy lại.

### Bước 4: Chạy full test (load model + render audio, ~1-2 phút)

```bash
python inference/download_female_ref.py ^
  --female-ref Models\references\female_ref.wav ^
  --male-ref D:\Documents\HUST\HUST_Project\Project_Final\data\StyleTTS2_preprocess\output_dataset\wavs\ngan_00002.wav
```

**Mong đợi output**:
```
HEALTH CHECK — FEMALE
  ✅ Health check PASS — file ready to use

HEALTH CHECK — MALE (Ngạn)
  ✅ Health check PASS — file ready to use

LOADING INFERENCE ENGINE
  ...
  ✅ Inference engine ready.

STYLE EXTRACT + SYNTHESIZE TEST — role=female
  ✅ Style shape: (1, 128)
     Mean: +0.0123  Std: 0.4567  Min: -1.2345  Max: +1.9876
  
[2/4] Kể chuyện (trung tính)
  Text: Đêm đó, cô gái trẻ lặng lẽ bước ra khỏi căn nhà cũ kỹ.
  Phn : ɗem˧ ɗo˨ʔ, ko˧ ɣaːj˧ ʈɛ̆˩ ...
  ✅ Saved: female_narration.wav  (3.45s, RTF=0.18x)

[3/4] Lời thoại nữ (hỏi)
  ...

[4/4] Cảm thán (sợ hãi)
  ...

STYLE EXTRACT + SYNTHESIZE TEST — role=male_ngan
  ... (tương tự, 3 câu nữa với giọng Ngạn)

TỔNG KẾT
FEMALE: ✅ Pass. 3 sample đã render.
MALE (Ngạn): ✅ Pass. 3 sample đã render.

Output folder: D:\...\TTS_StyleTTS2-lite-vi\output\female_ref_test
```

### Bước 5: Nghe đánh giá

Mở folder `output/female_ref_test/` và nghe 6 file `.wav` (3 female + 3 male). Quan sát:

1. **3 female_*.wav có thực sự là GIỌNG NỮ không?** Không phải giả nam, không méo.
2. **Phát âm tiếng Việt rõ ràng không?** Có lắp/nuốt từ không?
3. **3 câu có ngữ điệu khác nhau không?** (Kể chuyện flat, hỏi upward, cảm thán emphatic)
4. **So với male_ngan_*.wav**, female có DISTINCTIVELY khác giọng không? (Nếu giống y hệt giọng Ngạn → có vấn đề với style encoder, model bị overfit về Ngạn.)

---

## Tín hiệu báo lại cho tôi

**Sau khi chạy xong, gửi tôi:**

1. **Health check output** (copy log)
2. **Synthesize stats**: RTF của từng câu? Có sample nào fail không?
3. **Đánh giá audio**:
   - ✅ Giọng nữ rõ ràng + khác biệt với Ngạn → PASS → đi tiếp D2
   - ⚠️ Nghe được nhưng có issue (vd: hơi giống Ngạn, không rõ giới tính) → cần thử file nữ khác
   - ❌ Không phải giọng nữ / méo nặng / không nói được tiếng Việt → có vấn đề khác

Nếu kết quả ✅ → tôi viết **D2 (`nlp_generator.py`)** — Gemini 2.0 Flash API → `script.json` với phân vai narrator/character_male/character_female + `pause_after_ms`.

Lưu ý nhỏ: trước khi vào D2 bạn nên có sẵn **Gemini API key** (https://aistudio.google.com/apikey — 1 phút, miễn phí).