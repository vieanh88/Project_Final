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

# Tóm tắt file D1

## Điểm thiết kế chính

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

# Cách test D1

## Bước 1: Chuẩn bị file giọng nữ

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

## Bước 2: Copy D1 vào đúng folder

```
TTS_StyleTTS2-lite-vi/inference/download_female_ref.py
```

(Cùng folder với `inference_engine.py`.)

## Bước 3: Chạy chỉ health check trước (5 giây)

```bash
cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi

python inference/download_female_ref.py ^
  --female-ref Models\references\female_ref.wav ^
  --skip-synthesize
```

**Mong đợi**: Health check passes hoặc warnings nhẹ. Nếu errors → fix file rồi chạy lại.

## Bước 4: Chạy full test (load model + render audio, ~1-2 phút)

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

## Bước 5: Nghe đánh giá

Mở folder `output/female_ref_test/` và nghe 6 file `.wav` (3 female + 3 male). Quan sát:

1. **3 female_*.wav có thực sự là GIỌNG NỮ không?** Không phải giả nam, không méo.
2. **Phát âm tiếng Việt rõ ràng không?** Có lắp/nuốt từ không?
3. **3 câu có ngữ điệu khác nhau không?** (Kể chuyện flat, hỏi upward, cảm thán emphatic)
4. **So với male_ngan_*.wav**, female có DISTINCTIVELY khác giọng không? (Nếu giống y hệt giọng Ngạn → có vấn đề với style encoder, model bị overfit về Ngạn.)

---

# Tóm tắt file D2

## Điểm thiết kế chính

1. **SDK mới `google-genai`** (KHÔNG phải `google-generativeai` đã deprecated). Đã verify qua research là API hiện tại.

2. **Model `gemini-2.5-flash`** (không phải 2.0 Flash đã shutdown). Free tier: 1500 RPD, 10 RPM, 250K TPM — quá dư cho 1 truyện ma (~3-5 calls).

3. **Structured output qua Pydantic schema** — Gemini đảm bảo trả về JSON đúng format. **KHÔNG cần** parse markdown ```json``` hay regex extract. Validation hai tầng: server-side (Gemini) + client-side (Pydantic).

4. **Master prompt bằng tiếng Việt** với:
   - 5 quy tắc phân câu chi tiết
   - 4 quy tắc gán role (nhấn mạnh: nếu không chắc giới tính → mặc định `character_male` vì giọng chính là Ngạn)
   - 5 mức pause_after_ms (câu thường, câu nối, chuyển cảnh, cao trào, cuối đoạn) — match đúng tài liệu thiết kế ban đầu của bạn
   - 1 few-shot example đầy đủ

5. **Thinking mode tắt mặc định**: `gemini-2.5-flash` mặc định bật thinking → chậm gấp 2-3x cho task này (không cần reasoning sâu). Có flag `--no-thinking` (default) để override.

6. **Chunking cho truyện dài**: chia theo paragraph, không cắt giữa câu. Mặc định 8000 chars/chunk. Sau khi merge, re-assign ID liên tục từ 1.

7. **Retry exponential backoff**: 3 lần với delay 2s/4s/8s. Nếu 1 chunk fail hoàn toàn → skip + tiếp tục các chunk khác (không crash cả pipeline).

8. **2 output files**:
   - `script.json` — array các lines (đúng spec cho D3)
   - `script.metadata.json` — meta info riêng (model, timestamp, stats, ước tính duration audio)

9. **Dry-run mode**: in prompt sample không gọi API → bạn check prompt OK trước khi tốn quota.

---

# Cách test D2

## Bước 1: Chuẩn bị file truyện ma test

Tạo file `data/raw_stories/test_ghost_story.txt` (hoặc bất kỳ vị trí nào). Nội dung mẫu (200-500 từ là đủ để test):

```
Đêm hôm ấy, trời tối đen như mực. Cô gái trẻ tên Lan ngồi một mình trong căn nhà cũ kỹ ở ngoại ô Hà Nội. Bỗng nhiên, từ trong góc phòng, có giọng nói thì thầm:

"Đừng quay lại..."

Lan giật mình, tim đập thình thịch. Cô bước nhẹ về phía cửa, rồi gọi: "Ai đó? Có ai trong nhà không?"

Im lặng. Chỉ có tiếng gió rít qua khe cửa.

Bỗng một bàn tay lạnh ngắt chạm vào vai Lan. Cô quay lại, và thấy một bóng đen đứng sau lưng.

"Em ơi, anh đã chờ em rất lâu rồi..."
```

Lưu encoding UTF-8 (Notepad: File → Save As → Encoding: UTF-8).

## Bước 2: Cài deps

```bash
pip install google-genai python-dotenv pydantic
```

## Bước 3: Verify .env đã có API key

File `D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\.env`:
```
GEMINI_API_KEY=AQ.xxxxxxxxxxxxxxxxxx
```

## Bước 4: Dry-run trước để check prompt

```bash
cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi

python inference\nlp_generator.py ^
  --input data\raw_stories\test_ghost_story.txt ^
  --output output\scripts\test_ghost.json ^
  --dry-run
```

→ Sẽ in prompt đầy đủ ra terminal để bạn review. KHÔNG tốn API quota.

## Bước 5: Chạy thật

```bash
python inference\nlp_generator.py ^
  --input data\raw_stories\test_ghost_story.txt ^
  --output output\scripts\test_ghost.json
```

## Bước 6: Mong đợi output

```
============================================================
D2 — NLP GENERATOR (Gemini -> script.json)
============================================================
Input file  : D:\...\test_ghost_story.txt
Output file : D:\...\output\scripts\test_ghost.json
Model       : gemini-2.5-flash
Thinking    : ON (default)

[1/5] Reading input text...
  Length: 612 chars, 124 words

[2/5] Chunking (max 8000 chars/chunk)...
  1 chunk(s)

[3/5] Setup Gemini client...
  Loaded .env từ: D:\...\.env
  API key: AQ.xxxxx...xxxx (length 39)

[4/5] Calling Gemini API (1 chunks)...

  Chunk 1/1  (612 chars)
    [Attempt 1/3] Calling gemini-2.5-flash...
    ✅ Response trong 4.2s
    Sinh ra 12 script lines

[5/5] Post-process + save...

============================================================
✅ SUCCESS
============================================================
  Script  : D:\...\output\scripts\test_ghost.json
  Metadata: D:\...\output\scripts\test_ghost.metadata.json

  Stats:
    Lines           : 12
    Roles           : {'narrator': 8, 'character_female': 2, 'character_male': 2}
    Est. audio dur. : 56.4s (0.9 min)

  Preview 3 dòng đầu:
    [  1] [narrator          ] [1500ms] Đêm hôm ấy, trời tối đen như mực.
    [  2] [narrator          ] [ 800ms] Cô gái trẻ tên Lan ngồi một mình trong căn nhà cũ kỹ ở ngoại ô Hà Nội.
    [  3] [narrator          ] [ 800ms] Bỗng nhiên, từ trong góc phòng, có giọng nói thì thầm:
    ...
```

### Bước 7: Mở `script.json` để review

```json
[
  {
    "id": 1,
    "role": "narrator",
    "text": "Đêm hôm ấy, trời tối đen như mực.",
    "pause_after_ms": 1500
  },
  {
    "id": 2,
    "role": "narrator",
    "text": "Cô gái trẻ tên Lan ngồi một mình trong căn nhà cũ kỹ ở ngoại ô Hà Nội.",
    "pause_after_ms": 800
  },
  ...
  {
    "id": 4,
    "role": "character_male",
    "text": "Đừng quay lại...",
    "pause_after_ms": 2200
  },
  {
    "id": 6,
    "role": "character_female",
    "text": "Ai đó? Có ai trong nhà không?",
    "pause_after_ms": 800
  },
  ...
]
```

**Kiểm tra**:
- ID liên tục từ 1
- Role hợp lý (Lan = character_female, "anh đã chờ em" = character_male)
- Pause hợp lý cho từng câu
- Text không có chữ số/markdown

---

# Tóm tắt file D3

## Điểm thiết kế chính

1. **Class `AudiobookSynthesizer`** — wrapper sạch sẽ quanh engine:
   - Init 1 lần: load 2 styles vào dict `{role → style_tensor}`
   - `synthesize_line(text, role)` — render 1 câu
   - `synthesize_script(script)` — loop full script + chèn silence + concat

2. **Logic CHÍNH XÁC theo PDF gốc của bạn** (Phase 2 Bước 3):
   ```python
   for line in script:
       wav = engine.synthesize(text, style[role])
       audio_chunks.append(wav)
       silence = np.zeros(int(pause_after_ms/1000 * 24000), dtype=float32)
       audio_chunks.append(silence)
   final_wav = np.concatenate(audio_chunks)
   ```
   → Silence padding mathematical bằng `np.zeros` → KHÔNG có nhiễu/breath, đúng triết lý "tĩnh mịch chuẩn xác phim kinh dị".

3. **3 trạng thái xử lý line**:
   - `ok` — synthesize thành công, append wav
   - `skipped_empty` — text rỗng → chỉ chèn silence (giữ timing)
   - `failed` — exception → chèn 200ms silence ngắn + log error + tiếp tục (default behavior, có flag tắt)

4. **Memory hygiene cho 3050Ti 4GB**: `torch.cuda.empty_cache()` mỗi 10 lines. Audiobook 100 lines sẽ tự clean cache 10 lần.

5. **Speed per role configurable**: 
   - Default: tất cả 1.0
   - User có thể thử `--narrator-speed 0.9` cho dramatic kể chuyện
   - Engine D0 đã accept `speed` arg trong `synthesize()`

6. **3 output files**:
   - `audiobook.wav` — file chính, 24kHz mono float32
   - `audiobook.metadata.json` — metadata + per-line stats (đầy đủ debug info)
   - `audiobook_lines/line_XXXX_role.wav` — optional, từng line riêng nếu `--save-lines`

7. **Normalize peak về 0.95** (tránh clipping khi mix với silence). Có flag `--no-normalize` nếu cần raw.

8. **Progress bar tqdm** với postfix hiển thị id + role + thời gian — dễ debug nếu thấy 1 line tốn quá lâu.

9. **Validation đầu vào nghiêm ngặt**:
   - script.json phải là array (catch case user nhầm với .metadata.json)
   - Mỗi item phải có 4 fields `id, role, text, pause_after_ms`
   - 2 reference files phải tồn tại

10. **Summary cuối cùng**: in đầy đủ stats + role distribution + RTF + cảnh báo failed lines (top 10).

---

# Cách test D3

## Bước 1: Đảm bảo có đủ inputs

```
TTS_StyleTTS2-lite-vi/
├── output/scripts/test_ghost.json              ← từ D2
├── Models/checkpoints-v7-ep14/epoch_00013.pth  ← checkpoint fine-tuned
├── Models/references/female_ref.wav             ← từ D1 (file giọng nữ bạn cung cấp)
├── configs/config.yaml                          ← config gốc
└── inference/
    ├── inference_engine.py    ← D0
    └── tts_generator.py        ← D3 (vừa tạo)
```

Male ref dùng lại path trong dataset Ngạn đã clean:
```
D:\Documents\HUST\HUST_Project\Project_Final\data\StyleTTS2_preprocess\output_dataset\wavs\ngan_00002.wav
```

## Bước 2: Cài deps thêm

```bash
pip install tqdm
```

(Các deps khác đã có từ D0/D1/D2.)

## Bước 3: Chạy

```bash
cd D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi

python inference\tts_generator.py ^
  --script output\scripts\test_ghost.json ^
  --male-ref D:\Documents\HUST\HUST_Project\Project_Final\data\StyleTTS2_preprocess\output_dataset\wavs\ngan_00002.wav ^
  --female-ref Models\references\female_ref.wav ^
  --output output\audiobooks\test_ghost.wav ^
  --save-lines
```

(`--save-lines` để debug: nếu output cuối có chỗ kỳ, bạn nghe được từng line riêng để locate.)

## Bước 4: Mong đợi output

```
============================================================
D3 — TTS GENERATOR (Audiobook synthesis)
============================================================
Script      : ...\output\scripts\test_ghost.json
Male ref    : ...\ngan_00002.wav
Female ref  : ...\female_ref.wav
Output      : ...\output\audiobooks\test_ghost.wav
...

[1/5] Loading script...
  Loaded 12 lines

[2/5] Loading inference engine...
[INFO] Device       : cuda
[INFO] FP16 autocast: True
...
[INFO] ✅ Inference engine ready.

[3/5] Building AudiobookSynthesizer + computing styles...
  Computing male style from: ngan_00002.wav...
    shape=(1, 128), time=1.23s
  Computing female style from: female_ref.wav...
    shape=(1, 128), time=0.98s

[4/5] Synthesizing audiobook...

  Synthesizing 12 lines...
  Lines: 100%|████████████| 12/12 [00:08<00:00, id=12 narrator 0.6s]

  Concatenating 24 chunks...

[5/5] Saving outputs...
  Audiobook : ...\test_ghost.wav  (3.4 MB)
  Metadata  : ...\test_ghost.metadata.json
  Per-line  : ...\test_ghost_lines  (12 files)

  Lines total       : 12
    OK              : 12
    Failed          : 0
    Empty (skipped) : 0
  Roles distribution: {'narrator': 8, 'character_female': 2, 'character_male': 2}
  Total speech      : 36.5s
  Total audio out   : 56.2s (0.94 min)
  Total synth time  : 8.4s
  Overall RTF       : 0.230x (lower = faster than realtime)

============================================================
✅ AUDIOBOOK SYNTHESIS COMPLETE
============================================================
  Wall time   : 11.2s
  Audio dur   : 56.2s (0.94 min)
  Real-time x : 5.02x (audio_dur / synth_time)

👉 Mở file: ...\test_ghost.wav
```

## Bước 5: Nghe đánh giá

Mở `test_ghost.wav`. Đánh giá theo 5 tiêu chí:

1. **Mạch chảy tự nhiên**: speech → silence → speech có khớp không, có "cắt cụt" không?
2. **Khoảng lặng dramatic**: chỗ pause 1500-2500ms có cảm giác kinh dị tĩnh mịch không?
3. **Phân vai rõ ràng**: 
   - narrator (Ngạn) trầm, kể chuyện
   - character_male (Ngạn) cùng giọng nhưng có thể khác nhịp
   - character_female nghe rõ là giọng nữ
4. **Không artifacts**: không có click/pop, không có giọng vỡ giữa câu
5. **Tổng thể**: nghe như 1 audiobook horror tiếng Việt thực thụ chưa?

Nếu có chỗ kỳ → mở `test_ghost_lines/line_XXXX_role.wav` để debug line cụ thể.

---
