# **`monitor_training.py`** — script giám sát độc lập.

**Vị trí:** `TTS_StyleTTS2/fine-tune/monitor_training.py`

**Cài dependencies:**
```bash
pip install tensorboard requests python-dotenv
```

---

## Setup Discord Webhooks (2 thiết bị)

**Bước 1:** Thêm 2 webhook URL vào `.env` ở project root:
```
DISCORD_WEBHOOK_1=https://discord.com/api/webhooks/xxxxx/yyyyy
DISCORD_WEBHOOK_2=https://discord.com/api/webhooks/aaaaa/bbbbb
```

Bạn có thể tạo 2 webhook khác nhau (ví dụ: 1 cho channel trên PC, 1 cho channel riêng chỉ notification trên điện thoại), hoặc dùng cùng 1 URL cho cả 2 thiết bị.

**Bước 2:** Test webhook trước khi chạy thật:
```bash
python monitor_training.py --log-dir "Models/VietnameseBase" --stage 1 --dry-run
```
Dry-run sẽ print ra console mà không gửi Discord. Sau khi OK, bỏ `--dry-run`.

---

## Workflow chạy Stage 1 hoàn chỉnh

**Terminal 1 — Training:**
```bash
cd D:\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune
python train_wrapper.py --stage 1
```

**Terminal 2 — Monitor (chạy song song):**
```bash
cd D:\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune
python monitor_training.py --log-dir "Models/VietnameseBase" --stage 1
```

Các tham số cho đúng setup của bạn (đã là default):
- `--patience 10` (10 epochs liên tiếp không giảm đáng kể)
- `--min-delta 0.0005`
- `--progress-report-interval 20` (báo cáo mỗi 20 epochs)
- `--poll-interval 30` (scan TensorBoard mỗi 30s)

---

## Kiến trúc 3 lớp phát hiện

**Lớp 1 — MetricTracker** (một instance riêng cho mỗi metric):
- Track full history theo epoch
- So sánh với best value, `stale_count++` khi không cải thiện
- `increasing_count++` khi val loss tăng liên tiếp

**Lớp 2 — Handle flags** (mỗi lần nhận data mới):
- `is_nan=True` → gửi 🚨 CRITICAL ngay lập tức (training sẽ hỏng)
- `is_plateau=True` (chỉ 1 metric) → chỉ log ra console, chưa gửi Discord
- `increasing_count >= 5` → ⚠️ WARNING (overfitting của metric đó)

**Lớp 3 — Combined plateau** (điều kiện khó nhất):
- CHỈ gửi 🛑 CRITICAL khi **CẢ HAI** `eval/mel_loss` **VÀ** `eval/mono_align_loss` đều plateau → đây là signal đáng tin cậy nhất để bạn Ctrl+C

Mỗi loại cảnh báo chỉ gửi **1 lần duy nhất** (flag `plateau_alerted`, `overfit_alerted`) để không spam.

---

## Màu sắc notification trên Discord

- 🟦 **Xanh dương** (info) — Monitor start/stop
- 🟩 **Xanh lá** (progress) — Báo cáo tiến độ mỗi 20 epochs, kèm trend `📉 giảm / ➡️ ổn định / 📈 tăng`
- 🟧 **Cam** (warning) — Overfitting 1 metric
- 🟥 **Đỏ** (critical) — NaN hoặc CẢ HAI metrics plateau → đề xuất Ctrl+C

---

## Lưu ý quan trọng về tên metric

Script mặc định dùng tên tag `eval/mel_loss` và `eval/mono_align_loss`. Dựa trên code gốc StyleTTS2 tôi đã đọc, tag chính xác **có thể hơi khác** (ví dụ `eval/loss_mel`, `eval/align_loss`, hoặc viết hoa).

**Khi chạy monitor lần đầu, script sẽ in toàn bộ tags có sẵn trong events file** (xem log `TensorBoard tags có sẵn`). Nếu thấy tên metric khác với default, chỉ cần override:
```bash
python monitor_training.py --log-dir ... --stage 1 \
    --primary-metric "eval/loss_mel" \
    --secondary-metric "eval/align_loss"
```

Tôi sẽ sửa file theo thứ tự:

1. **Bug 1**: Đổi default `secondary_metric` thành None, set theo stage trong main()
2. **Bug 2**: `_handle_flags` thiếu alert plateau cá nhân → thêm
3. **Bug 3**: `datetime.utcnow()` deprecated → đổi sang `datetime.now(timezone.utc)`
4. **Wandb integration**: lazy import + 3 namespace define_metric + log scalars + alerts + cleanup
5. **CLI flags mới**: `--no-discord`, `--wandb-project`, `--wandb-run-name`, `--wandb-resume`
6. **Comments**: USER CUSTOMIZE markers cho run_name template + tags

## Chỉnh sửa thêm
### 3 Bug đã fix

| Bug | Trước | Sau |
|---|---|---|
| **#1** secondary_metric `eval/mono_align_loss` không tồn tại | KHÔNG BAO GIỜ trigger combined plateau | Auto-set theo stage: Stage 1 → `train/mel_loss`, Stage 2/3 → `eval/dur_loss` |
| **#2** `_handle_flags` không trigger plateau alert cá nhân | Tracker set flag nhưng nobody gọi broadcast | Plateau alert ngay khi từng metric plateau (warning level) |
| **#3** `datetime.utcnow()` deprecated từ Python 3.12 | DeprecationWarning trên Vast.ai instance | `datetime.now(timezone.utc)` chuẩn |

### Wandb integration (Option A' của bạn)

**`WandbManager` class** với lifecycle đầy đủ:
- `__init__()` — lazy, không import wandb
- `init()` — lazy import wandb, call `define_metric()` cho **3 namespace**, không crash nếu fail
- `log_scalars()` — log mỗi entry với step bind đúng namespace, **KHÔNG truyền `step=`**
- `log_monitor_state()` — log `monitor/poll`, `monitor/{tag}_best`, `_stale`, `_increasing`
- `alert()` — trigger `wandb.alert()` với level mapping (critical→ERROR, warning→WARN, info/progress→INFO)
- `finish()` — cleanup với `exit_code` từ KeyboardInterrupt / exception

**3 namespace với `define_metric()`**:
```python
wandb.define_metric("train/iter")
wandb.define_metric("train/*", step_metric="train/iter")   # iters

wandb.define_metric("eval/epoch")
wandb.define_metric("eval/*", step_metric="eval/epoch")    # epoch

wandb.define_metric("monitor/poll")
wandb.define_metric("monitor/*", step_metric="monitor/poll")  # poll counter
```

Wandb dashboard tự dùng đúng X-axis cho mỗi namespace, user KHÔNG cần edit gì trong UI.

### CLI flags mới

| Flag | Mục đích |
|---|---|
| `--no-discord` | Tắt Discord (chỉ wandb) |
| `--wandb-project NAME` | Override project (default `story-ai-narrator`) |
| `--wandb-run-name NAME` | Override run name (default auto-gen từ template) |
| `--wandb-resume RUN_ID` | Resume wandb run cũ (default tạo run mới) |
| `--wandb-tags "a,b,c"` | Override tags (default `[styletts2,vietnamese,vivoice]`) |

### USER CUSTOMIZE markers (đầu file)

```python
# === USER CUSTOMIZE — Run name template ===
WANDB_RUN_NAME_TEMPLATE = "{stage_short}_{timestamp}"  # ← USER CUSTOMIZE

# === USER CUSTOMIZE — Tags ===
WANDB_DEFAULT_TAGS = ["styletts2", "vietnamese", "vivoice"]  # ← USER CUSTOMIZE

# === USER CUSTOMIZE — Project name ===
WANDB_DEFAULT_PROJECT = "story-ai-narrator"  # ← USER CUSTOMIZE
```

### Cách dùng

```bash
# Set env vars (1 lần / session)
export WANDB_API_KEY="..."
export DISCORD_WEBHOOK_1="https://discord.com/api/webhooks/..."

# Default: cả wandb + Discord (nếu có cả 2 secrets)
python monitor_training.py --log-dir Models/VietnameseBase --stage 1

# Chỉ wandb (tắt Discord)
python monitor_training.py --log-dir Models/VietnameseBase --stage 1 --no-discord

# Resume wandb run cũ
python monitor_training.py --log-dir Models/VietnameseBase --stage 1 \
    --wandb-resume abc123xyz

# Dry-run (không gửi gì)
python monitor_training.py --log-dir Models/VietnameseBase --stage 1 --dry-run
```

## Giải thích chi tiết các loss

### Stage 1 — `train_first.py` (Acoustic & Alignment)

| Loss | Tag | Ý nghĩa | Khi nào active | Hành vi mong đợi |
|---|---|---|---|---|
| **mel_loss** | `train/mel_loss`, `eval/mel_loss` | L1 distance giữa mel-spectrogram dự đoán và ground truth. **Loss QUAN TRỌNG NHẤT** — đo độ chính xác về âm thanh. | Mọi epoch | Giảm đều từ ~3.0 → ~0.4 trong vài epoch đầu, sau đó giảm chậm về ~0.3 |
| **gen_loss** | `train/gen_loss` | Adversarial loss cho Generator (MPD + MSD). Khi gen thắng discriminator → giá trị thấp. | Mọi epoch | Dao động quanh ~1.0-2.0. KHÔNG cần giảm xuống 0 (đó là dấu hiệu mode collapse). |
| **d_loss** | `train/d_loss` | Discriminator loss (MPD + MSD đánh giá audio thật/giả). Khi disc thắng → giá trị thấp. | Mọi epoch | Cùng range với gen_loss, gen và disc cân bằng nhau. |
| **mono_loss** | `train/mono_loss` | Monotonic Alignment Loss — buộc alignment phoneme → mel phải monotonic (đi từ trái sang phải, không nhảy lung tung). | TỪ epoch `TMA_epoch` trở đi (config bạn = 10) | Trước epoch 10: = 0. Sau epoch 10: giảm dần từ ~5.0 → ~1.0 |
| **s2s_loss** | `train/s2s_loss` | Sequence-to-Sequence loss của text aligner ASR (dự đoán phoneme từ mel) — giúp model học attention chính xác. | TỪ epoch `TMA_epoch` trở đi | Cùng pattern với mono_loss |
| **slm_loss** | `train/slm_loss` | Speech Language Model loss (WavLM feature matching) | **KHÔNG ACTIVE ở Stage 1** (`joint_epoch=9999`). Sẽ ghi giá trị = 0 hoặc dummy. | = 0 suốt Stage 1 |

### Stage 2 — `train_second.py` (Expressive Training)

Bao gồm tất cả ở Stage 1 + thêm các loss expressive:

| Loss | Tag | Ý nghĩa | Khi nào active |
|---|---|---|---|
| **ce_loss** | `train/ce_loss` | Cross-Entropy loss của duration predictor (dự đoán độ dài mỗi phoneme). | Toàn bộ Stage 2 |
| **dur_loss** | `train/dur_loss`, `eval/dur_loss` | Duration loss — L1 distance giữa duration dự đoán và ground truth. | Toàn bộ Stage 2 |
| **norm_loss** | `train/norm_loss` | Norm consistency — giữ năng lượng audio nhất quán giữa GT và predicted. | Toàn bộ Stage 2 |
| **F0_loss** | `train/F0_loss`, `eval/F0_loss` | F0 (pitch) reconstruction loss — đo accuracy của pitch contour. | Toàn bộ Stage 2 |
| **sty_loss** | `train/sty_loss` | Style reconstruction loss — buộc style vector tái tạo được từ audio. | Toàn bộ Stage 2 |
| **diff_loss** | `train/diff_loss` | Style **diffusion** loss (score matching). | TỪ epoch `diff_epoch` (config bạn = 5) |
| **d_loss_slm** | `train/d_loss_slm` | SLM Discriminator (WavLM-based) — đánh giá "giống người" hay không. | TỪ epoch `joint_epoch` (config bạn = 10) |
| **gen_loss_slm** | `train/gen_loss_slm` | SLM Generator loss đối ứng với d_loss_slm. | TỪ epoch `joint_epoch` |
| **slm_loss** | `train/slm_loss` | SLM feature matching (khác `gen_loss_slm`). | Toàn bộ Stage 2 |

**LƯU Ý CRITICAL**: Stage 2 KHÔNG có `train/mono_loss` và `train/s2s_loss` — tags này CHỈ tồn tại ở Stage 1.

### Eval metrics (validation set)

- **Stage 1**: chỉ có `eval/mel_loss` (DUY NHẤT)
- **Stage 2**: có `eval/mel_loss`, `eval/dur_loss`, `eval/F0_loss`
