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
