# **D4: `train_wrapper.py`** — bộ não điều phối toàn bộ pipeline huấn luyện.

**Đặt vào:** `TTS_StyleTTS2/fine-tune/train_wrapper.py`

**Cách sử dụng:**
```bash
# Stage 1: Acoustic & Alignment
python train_wrapper.py --stage 1

# Stage 2: Expressive Training (auto-load checkpoint từ Stage 1)
python train_wrapper.py --stage 2

# Stage 3: Fine-tune Bác Ngạn (auto-load checkpoint từ Stage 2)
python train_wrapper.py --stage 3

# Chỉ kiểm tra config mà không chạy:
python train_wrapper.py --stage 1 --dry-run

# Override thủ công:
python train_wrapper.py --stage 2 \
    --pretrained-model "path/to/specific_checkpoint.pth" \
    --batch-size 4 \
    --vocab-file "path/to/phoneme_vocab.json"
```

**3 chức năng cốt lõi:**

**1. Vocab Injection** — Đọc `phoneme_vocab.json`, lấy `n_token` thực tế (ví dụ: 195), rồi thay thế placeholder `178` trong `model_params.n_token` của config trước khi gọi script gốc. Điều này chống lỗi `IndexError` 100% khi embedding layer nhận index vượt quá vocab size.

**2. Auto-Chain Checkpoint** — Khi chạy Stage 2, wrapper tự quét `log_dir` của Stage 1 (`Models/VietnameseBase/`), tìm file `.pth` có tên chứa "best" hoặc mới nhất theo thời gian, rồi điền vào `pretrained_model`. Tương tự Stage 3 quét từ Stage 2. Bạn luôn có thể override bằng `--pretrained-model`.

**3. Subprocess Delegation** — Wrapper KHÔNG viết lại training loop (quá rủi ro với MAS + JAT). Thay vào đó, nó lưu config đã xử lý ra `config/_processed/config_stageX_processed.yaml`, rồi gọi `subprocess.Popen` tới `train_first.py` / `train_second.py` / `train_finetune.py` gốc, stream output realtime ra console + log file.

**Pre-flight checks** trước mỗi lần chạy: kiểm tra CUDA, GPU VRAM, script gốc tồn tại, config template tồn tại, vocab file tồn tại, data files tồn tại. Nếu có lỗi → dừng ngay, không lãng phí thời gian.

**Auto-infer paths:** Nếu không chỉ định, wrapper tự suy `styletts2_root = ../StyleTTS2`, `finetune_root = thư mục chứa train_wrapper.py`, `vocab_file` tìm trong `data_pipeline/prepare_vicoice/output/`.

## Phân tích & Kế hoạch thiết kế

### Câu hỏi thiết kế: Stage 1 resume hoạt động như thế nào?

Trong `train_first.py` của repo gốc StyleTTS2, logic load checkpoint là:

```python
if pretrained_model != "":
    model, optimizer, start_epoch, iters = load_checkpoint(
        model, optimizer, pretrained_model,
        load_only_params=config.get('load_only_params', False)
    )
else:
    start_epoch = 0
    iters = 0
```

**Khi resume**: cần `pretrained_model = <latest epoch_1st_*.pth>` + `load_only_params = false` để load CẢ `optimizer state` + `start_epoch` + `iters` → training tiếp tục đúng chỗ.

### 3 tình huống Stage 1 cần xử lý

| Tình huống | Cách detect | Hành vi |
|---|---|---|
| **Lần đầu** | `log_dir` chưa tồn tại HOẶC không có `epoch_1st_*.pth` | `pretrained_model=""` (train từ đầu) |
| **Resume** | Có `epoch_1st_*.pth` trong `log_dir` | `pretrained_model=<latest>` + `load_only_params=false` |
| **Override** | User truyền `--pretrained-model` | Tin user (giống logic Stage 2/3 đang có) |

### Tích hợp với code hiện tại

Code hiện tại có sẵn pattern `_detect_stage2_resume(stage2_log_dir)`. Tôi sẽ:

1. **Tạo function mới** `_detect_stage1_resume()` — đối xứng với `_detect_stage2_resume()`
2. **Sửa branch `if stage == 1`** trong `auto_chain_checkpoint()` để gọi detect function trước khi set `pretrained_model=""`
3. **Vị trí override**: hiện tại logic `if override_pretrained` nằm SAU `if stage == 1`. Cần dời lên TRƯỚC để Stage 1 cũng có thể nhận override (hữu ích nếu user muốn warm-start từ ckpt cụ thể).

### Edge case quan trọng

**E1**: Nếu Stage 1 đã train xong (epoch cuối = epochs_1st) → file ckpt cuối là `first_stage.pth` (lưu khi finalize), KHÔNG phải `epoch_1st_XXX.pth`. Khi resume, ưu tiên:
1. `epoch_1st_*.pth` mới nhất (resume từ giữa chừng)
2. Fallback `first_stage.pth` (nếu đã train xong và muốn tiếp tục)

→ Tôi sẽ làm theo thứ tự ưu tiên này.

**E2**: Nếu user resume nhưng config `epochs_1st` ≤ start_epoch → `train_first.py` vẫn chạy nhưng for-loop epoch không lặp lần nào → output rỗng. Tôi sẽ ADD warning log cảnh báo user check số epoch.

**E3**: `first_stage_path` field trong config_stage1.yaml — đây là tên file để train_first.py LƯU ra ở cuối Stage 1 (KHÔNG phải để load). Không cần đụng tới.

### Thay đổi từng phần

Thêm 3 thay đổi (concise, ít risk):

1. Thêm function `_detect_stage1_resume()` ngay sau `_detect_stage2_resume()`
2. Sửa branch `if stage == 1` trong `auto_chain_checkpoint()` 
3. Dời `if override_pretrained` lên trước cả `if stage == 1` để áp dụng được cho mọi stage

### 3 thay đổi chính (tổng +115 lines, không xoá logic cũ)

**1. Thêm function `_detect_stage1_resume()`** (đối xứng với `_detect_stage2_resume()`):
- Tìm `epoch_1st_*.pth` mới nhất theo mtime (ưu tiên 1)
- Fallback `first_stage.pth` nếu Stage 1 đã train xong (ưu tiên 2, có warning vì file này KHÔNG có optimizer state)
- Return None nếu log_dir không tồn tại hoặc không có ckpt

**2. Sửa branch `if stage == 1`** trong `auto_chain_checkpoint()`:
- Resolve log_dir Stage 1 thành absolute path
- Gọi `_detect_stage1_resume()` trước khi quyết định:
  - Có ckpt → RESUME: set `pretrained_model = <latest>` + `load_only_params = false`
  - Không có ckpt → train từ đầu: `pretrained_model = ""`
- Thêm WARNING khi resume từ ckpt gần `epochs_1st` (vd: ckpt epoch 29, config epochs_1st=30 → cảnh báo "có thể đã train xong, tăng epochs_1st")

**3. Dời `if override_pretrained` lên TRƯỚC `if stage == 1`**:
- Áp dụng cho mọi stage (Stage 1/2/3) thay vì chỉ Stage 2/3
- Stage 1 override: chỉ set `pretrained_model` + `load_only_params=false`, KHÔNG set `second_stage_load_pretrained` (field này chỉ Stage 2/3 dùng)
- Stage 2/3 override: giữ logic cũ (set `second_stage_load_pretrained=True` cho PATH B)

### 3 use cases mới sau khi sửa

| Use case | Cách kích hoạt | Hành vi |
|---|---|---|
| **Stage 1 lần đầu** | `python train_wrapper.py --stage 1` (log_dir rỗng) | `pretrained_model=""` → train từ epoch 0 |
| **Stage 1 RESUME** | `python train_wrapper.py --stage 1` (có sẵn `epoch_1st_*.pth`) | Auto pick latest, training tiếp tục đúng chỗ |
| **Stage 1 OVERRIDE** | `python train_wrapper.py --stage 1 --pretrained-model "path/epoch_1st_00015.pth"` | Dùng ckpt cụ thể (skip auto-detect) |
