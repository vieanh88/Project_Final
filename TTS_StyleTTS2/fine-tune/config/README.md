# **D1-D3: 3 file config stage**.

---

**Tóm tắt sự khác biệt giữa 3 giai đoạn:**

| Tham số | Stage 1 | Stage 2 | Stage 3 |
|---------|---------|---------|---------|
| **Script gốc** | `train_first.py` | `train_second.py` | `train_finetune.py` |
| **Schema** | `config.yml` | `config.yml` | `config_ft.yml` |
| **Dataset** | ViVoice | ViVoice | Bác Ngạn |
| **OOD Text** | ❌ Không | ✅ 50k câu truyện ma | ✅ Giữ nguyên |
| **JAT/SLM** | ❌ `joint_epoch: 9999` | ✅ `joint_epoch: 0` | ✅ `joint_epoch: 0` |
| **Batch size** | 16 (VRAM dư dả) | 4 (JAT tốn VRAM) | 6 |
| **LR (ft_lr)** | 1e-5 | 1e-5 | **1e-5** (chống forgetting) |
| **Decoder** | iSTFTNet | iSTFTNet | iSTFTNet |
| **multispeaker** | **true** | **true** | **true** |
| **load_only_params** | false | false | **true** (reset optimizer) |
| **pretrained_model** | (trống) | ← auto từ Stage 1 | ← auto từ Stage 2 |

**3 điểm thiết kế cốt lõi:**

Thứ nhất, `n_token: 150` trong cả 3 file là **placeholder**. `train_wrapper.py` (file tiếp theo) sẽ đọc `phoneme_vocab.json`, lấy giá trị `n_token` thực tế (thực tế đang là 150), và tự động inject vào trước khi gọi script gốc. Điều này chống lỗi `IndexError` 100%.

Thứ hai, `pretrained_model: ""` ở Stage 2 và 3 cũng là placeholder. Wrapper sẽ quét thư mục log của giai đoạn trước, tìm checkpoint tốt nhất, và tự điền đường dẫn.

Thứ ba, Stage 3 dùng schema khác (`config_ft.yml`) nên có vài điểm khác biệt: chỉ có trường `epochs` (không tách `epochs_1st/epochs_2nd`), `multispeaker: true`, decoder `istftnet`, và `load_only_params: true` để reset optimizer state khi chuyển sang domain mới.

---
