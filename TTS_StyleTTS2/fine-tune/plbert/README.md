# **C1: `plbert/step1_build_corpus.py`**.

**Đặt vào:** `TTS_StyleTTS2/fine-tune/plbert/step1_build_corpus.py`

**Chạy:**
```bash
# Cách 1: CLI flags (linh hoạt, không cần config)
python step1_build_corpus.py \
    --vivoice-train "../data_pipeline/prepare_vicoice/output/vivoice_train_list.txt" \
    --vivoice-val   "../data_pipeline/prepare_vicoice/output/vivoice_val_list.txt" \
    --ngan-train    "../data_pipeline/prepare_ngan/output/ngan_train_list.txt" \
    --ngan-val      "../data_pipeline/prepare_ngan/output/ngan_val_list.txt" \
    --ood           "../data_pipeline/prepare_ood/output/OOD_texts_phoneme.txt"

# Cách 2: Config YAML
python step1_build_corpus.py --config config.yaml
```

**Section config cần thêm (nếu dùng cách 2):**
```yaml
plbert:
  filelist_sources:
    - "path/to/vivoice_train_list.txt"
    - "path/to/vivoice_val_list.txt"
    - "path/to/ngan_train_list.txt"
    - "path/to/ngan_val_list.txt"
  ood_sources:
    - "path/to/OOD_texts_phoneme.txt"
  corpus_file: "all_corpus_phoneme.txt"
  deduplicate: true
  shuffle: true
```

**Script này làm gì:**

Gộp phoneme text từ 3 nguồn hoàn toàn khác format. Với filelist (ViVoice, Ngạn) format `wav_path|phoneme|speaker_id`, nó trích riêng cột phoneme (index 1). Với OOD format chỉ có phoneme thuần, nó đọc thẳng. Sau đó lọc theo độ dài, deduplicate (loại dòng trùng lặp — quan trọng vì ViVoice có thể có câu giống nhau), shuffle toàn bộ, rồi lưu thành `all_corpus_phoneme.txt`.

Cuối cùng lưu `corpus_stats.json` chứa thống kê chi tiết: tổng dòng, tổng ký tự, phân bổ theo từng nguồn (bao nhiêu % đến từ ViVoice, bao nhiêu từ Ngạn, bao nhiêu từ OOD). File stats này hữu ích để verify tỷ lệ data trước khi train PL-BERT.

# **C2: `plbert/step2_train_plbert.py`**.

**Đặt vào:** `TTS_StyleTTS2/fine-tune/plbert/step2_train_plbert.py`

**Cài thêm dependencies:**
```bash
pip install torch python-dotenv
```

**Chạy:**
```bash
# Cách 1: CLI flags
python step2_train_plbert.py \
    --corpus "output/all_corpus_phoneme.txt" \
    --vocab  "output/phoneme_vocab.json" \
    --output-dir "./plbert_checkpoints" \
    --epochs 20 \
    --batch-size 64

# Cách 2: Config YAML
python step2_train_plbert.py --config config.yaml
```

**Section config (nếu dùng YAML):**
```yaml
plbert_train:
  corpus_file: "output/all_corpus_phoneme.txt"
  vocab_file: "output/phoneme_vocab.json"
  output_dir: "./plbert_checkpoints"
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  epochs: 20
  batch_size: 64
  learning_rate: 0.0001
  warmup_steps: 1000
  mlm_probability: 0.15
  fp16: false
  save_freq: 2
```

**Kiến trúc script:**

**PhonemeTokenizer** — Đọc `phoneme_vocab.json`, encode/decode ở cấp character-level, tự động thêm `<mask>` token nếu chưa có trong vocab.

**PhonemeMLMDataset** — Mỗi dòng corpus = 1 sequence phoneme. Encode → pad → apply MLM masking chuẩn BERT (15% tokens: 80% thay `[MASK]`, 10% random, 10% giữ nguyên).

**PLBERTModel** — BERT model dùng `nn.TransformerEncoder` với Pre-LN (ổn định hơn khi train from scratch). Có MLM head ở trên cùng. Cấu hình mặc định 768 hidden / 12 layers / 12 heads — khớp với kỳ vọng của StyleTTS2 khi load PL-BERT.

**Training loop** — Theo phong cách RT-DETR: cosine scheduler + linear warmup, gradient clipping, mixed precision (tùy chọn), log chi tiết mỗi N steps. Checkpoint lưu format `.t7` tương thích StyleTTS2 (`{"net": state_dict, ...}`). Auto-cleanup giữ lại N checkpoints mới nhất + best.

Output folder `plbert_checkpoints/` chứa `plbert_vi_best.t7` + `plbert_config.json`, trỏ vào `PLBERT_dir` trong config YAML của StyleTTS2.

---

## Hướng dẫn setup Wandb

**1. Cài đặt & đăng nhập (chạy 1 lần duy nhất):**
```bash
pip install wandb

# Đăng nhập — mở link, copy API key, paste vào terminal
wandb login
```
Hoặc lưu API key vào file `.env`:
```
WANDB_API_KEY=your_api_key_here
```

**2. Cấu trúc quản lý trên wandb:**

```
wandb.ai/<username>/
  └── ghost-story-narrator          ← PROJECT (gom toàn bộ dự án)
        ├── plbert-vi-base-ep20-bs64       ← RUN 1 (phiên chạy đầu tiên)
        ├── plbert-vi-base-ep20-bs32-fp16  ← RUN 2 (thử batch nhỏ hơn)
        ├── plbert-vi-large-ep10           ← RUN 3 (thử model lớn hơn)
        └── ...
```

**3. Cách đặt tên trong config:**

Trong `config_step2.yaml`, phần `wandb:` có 3 trường quan trọng:

**`project`** — Tên project, gom tất cả phiên chạy liên quan. Tôi đặt `"ghost-story-narrator"` để sau này khi train StyleTTS2 Stage 1-2-3, bạn cũng có thể log vào cùng project này. Nếu muốn tách riêng PL-BERT, đổi thành `"plbert-vietnamese"`.

**`run_name`** — Tên phiên chạy cụ thể. Quy ước đặt tên tôi khuyến nghị: `<model>-<ngôn ngữ>-<thay đổi chính>`. Ví dụ:
  - `plbert-vi-base-ep20-bs64` — lần chạy đầu tiên
  - `plbert-vi-base-ep20-bs32-fp16` — nếu bạn thử giảm batch + bật fp16
  - `plbert-vi-base-ep30-warmup4k` — nếu tăng epoch + warmup

**`tags`** — Dùng để filter nhanh trên dashboard. Tôi đặt `"plbert,vietnamese,mlm,stage-c2"` để sau này bạn có thể filter chỉ xem các run PL-BERT, hoặc chỉ xem stage-c2.

**4. Chạy training:**
```bash
python step2_train_plbert.py --config config_step2.yaml
```

**5. Theo dõi realtime:**

Ngay khi training bắt đầu, terminal sẽ in ra URL dạng:
```
Wandb initialized: project='ghost-story-narrator', run='plbert-vi-base-ep20-bs64', url=https://wandb.ai/xxx/ghost-story-narrator/runs/abc123
```
Mở link đó trên trình duyệt. Bạn sẽ thấy dashboard realtime với các biểu đồ: `train/loss` (step-level), `epoch/val_loss` (epoch-level), `train/learning_rate` (cosine curve), `train/speed_steps_per_sec`, và gradient/weight distributions (mỗi 500 steps).

**6. Tắt wandb (nếu cần):**

Đặt `enabled: false` trong config hoặc chạy offline:
```bash
WANDB_MODE=offline python step2_train_plbert.py --config config_step2.yaml
```