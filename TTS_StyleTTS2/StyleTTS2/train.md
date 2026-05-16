# Giải thích chi tiết các loss

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
