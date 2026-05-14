# Roadmap cập nhật (21 files)

| # | File | Mô tả |
|---|------|-------|
| A1 | `step1_rephonemize_lite.py` | Phonemize text Ngạn bằng viphoneme + replace _ → space + normalize whitespace + skip lỗi G2P |
| A2 | `step2_make_filelist_lite.py` | (1) Validate từng ký tự thuộc 189 vocab, (2) normalize backslash → forward slash, (3) filter audio < 0.5s, (4) ghi train.txt + val.txt đúng format 2 cột |
| A3 | `step3_zero_shot_test.ipynb` | Notebook test zero-shot trên 3-5 sample Ngạn (dùng pretrained chưa fine-tune) để verify phoneme tương thích |
| B1 | `KAGGLE_SETUP.md` | Hướng dẫn upload dataset, settings notebook, lưu ý quota |
| B2 | `kaggle_finetune_ngan.ipynb` | Notebook training: clone repo, install deps, mount data, override config, train, monitor, save |
| C1 | `config_ngan_kaggle.yml` | Config override (batch_size, max_len, ft_lr, freeze_modules=['style_encoder'], pretrained_model path Kaggle) |
| D1 | `download_female_ref.py` | Trích 1 đoạn audio nữ chất lượng cao từ ViVoice HF + slot để bạn drop file riêng |
| D2 | `nlp_generator.py` | Phase 1 với Gemini 2.0 Flash API → script.json với role ∈ {narrator, character_male, character_female} |
| D3 | `tts_generator.py` | Phase 2: load 2 ref audio + checkpoint Ngạn → loop sinh audio + silence padding np.zeros() → export wav |
| E1 | `evaluate_demo.py` | Render 5 sample đối sánh: pretrained zero-shot vs fine-tuned (cho phần MOS đồ án) |

# Cấu trúc thư mục

```
HUST_Project/Project_Final/
├── TTS_StyleTTS2-lite-vi_preprocess/    ← pipeline cũ (đã chạy xong)
│   └── output_dataset/
│       ├── wavs/
│       │   ├── ngan_00001.wav
│       │   └── ...
│       ├── filelist_train.txt           ← input cho file A1 (text gốc)
│       └── filelist_val.txt             ← input cho file A1 (text gốc)
│
└── TTS_StyleTTS2-lite-vi/               ← project mới (sẽ tạo)
    ├── data_pipeline/
    │   └── prepare_ngan_lite/
    │       ├── step1_rephonemize_lite.py        ← FILE A1
    │       ├── step2_make_filelist_lite.py      ← FILE A2
    │       └── step3_zero_shot_test.ipynb       ← FILE A3
    ├── configs/
    │   └── config_ngan_kaggle.yml               ← FILE C1
    ├── kaggle_notebooks/
    │   └── kaggle_finetune_ngan.ipynb           ← FILE B2
    ├── inference/
    │   ├── nlp_generator.py                     ← FILE D2
    │   ├── tts_generator.py                     ← FILE D3
    │   ├── download_female_ref.py               ← FILE D1
    │   └── evaluate_demo.py                     ← FILE E1
    ├── output/                                  ← chứa filelist_*_lite.txt sau khi A2 chạy
    │   ├── ngan_train_lite.txt
    │   └── ngan_val_lite.txt
    └── KAGGLE_SETUP.md                          ← FILE B1
```
