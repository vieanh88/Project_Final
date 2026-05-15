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
