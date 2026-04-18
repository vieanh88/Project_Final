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

---

**Nhóm D hoàn tất!** Tiến độ:

| Nhóm | Status |
|------|--------|
| A-C: Data Pipeline + PL-BERT (11 files) | ✅ Done |
| D: Training configs + wrapper (4 files) | ✅ Done |
| E: Inference (3 files) | ⏳ Tiếp theo |
