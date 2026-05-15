## Roadmap cập nhật (21 files)

**A. Data Preparation — ViVoice (5 scripts + 1 config)**

| # | File | Mô tả |
|---|------|-------|
| A0 | `prepare_vivoice/config.yaml` | Config chung cho folder này |
| A1 | `prepare_vivoice/step1_download.py` | Tải parquet từ HF về local cache |
| A2 | `prepare_vivoice/step2_extract_audio.py` | Decode bytes → resample 24kHz mono 16-bit → lưu .wav |
| A3 | `prepare_vivoice/step3_phonemize.py` | Text thô → phoneme (gọi logic từ `step08_phonemize.py` của bạn) |
| A4 | `prepare_vivoice/step4_build_vocab.py` | Quét toàn bộ phoneme → `phoneme_vocab.json` + `n_token` |
| A5 | `prepare_vivoice/step5_make_filelist.py` | Gộp wav + phoneme + speaker_id → train/val filelist |
| M1 | `prepare_vivoice/step6_rebuild_multispeaker` | Mapping channel ↔ speaker_id + stats |
| M2 | `prepare_vivoice/step6b_apply_filters` | Lọc speaker_id trực tiếp từ folder clean_wavs thay vì hf_cache |

**B. Data Preparation — Ngạn & OOD (3 scripts)**

| # | File | Mô tả |
|---|------|-------|
| B1 | `prepare_ngan/step1_phonemize.py` | filelist Ngạn (text thô) → phoneme, clean ký tự đặc biệt/số |
| B2 | `prepare_ngan/step2_make_filelist.py` | Append `\|0`, split train/val |
| B3 | `prepare_ood/step1_clean_phonemize.py` | Clean + phonemize 50k câu OOD |

**C. PL-BERT tiếng Việt (2 scripts)**

| # | File | Mô tả |
|---|------|-------|
| C1 | `plbert_v2/step1_build_corpus.py` | Gộp phoneme text từ ViVoice + Ngạn + OOD → `all_corpus_phoneme.txt` |
| C2 | `plbert_v2/step2_train_plbert.py` | Train PL-BERT từ đầu với vocab mới |
| C3 | `plbert_v2/verify_plbert_checkpoint.py` | Copy Utils/PLBERT/util.py của StyleTTS2 để kiểm tra checkpoint |

**D. Training Configs & Wrapper (4 files)**

| # | File | Mô tả |
|---|------|-------|
| D1 | `configs/config_stage1.yaml` | GĐ1: Acoustic (tắt JAT, `joint_epoch: 9999`) |
| D2 | `configs/config_stage2.yaml` | GĐ2: Expressive (`joint_epoch: 0`, OOD text, batch nhỏ) |
| D3 | `configs/config_stage3.yaml` | GĐ3: Fine-tune Ngạn (`load_only_params: true`, LR 1e-5) |
| D4 | `train_wrapper.py` | Nhạc trưởng: inject n_token, auto-chain checkpoint, gọi subprocess |

**E. Inference (3 files)**

| # | File | Mô tả |
|---|------|-------|
| E1 | `nlp_generator.py` | Phase 1: Truyện .txt → Qwen API → `script.json` |
| E2 | `create_mean_style.py` | Trích xuất mean style vector → `ngan_mean_style.pt` |
| E3 | `tts_generator.py` | Phase 2: JSON → phonemize → TTS → silence padding → export .wav |
| E4 | `monitor_training.py` | Script giám sát độc lập gửi thông báo qua discord webhook |