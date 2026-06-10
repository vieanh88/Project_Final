# Audiobook Studio — Demo UI

UI demo cho **StyleTTS2-lite-vi**: biến văn bản truyện ma tiếng Việt thành audiobook.
FastAPI (backend) + HTML/CSS/JS thuần (frontend), dark mode tông tím Indigo.

## Cấu trúc

```
webui/
├── backend/
│   ├── app.py          # FastAPI: routes, lifespan (load engine 1 lần), static mounts
│   ├── pipeline.py     # Cầu nối với code inference/ (engine, style cache, synth, NLP)
│   ├── jobs.py         # Job nền + theo dõi tiến độ cho tổng hợp audiobook
│   └── runtime/        # (tự sinh) uploads/ + audio/ — đã gitignore
├── frontend/
│   ├── index.html
│   ├── css/styles.css
│   └── js/app.js
└── requirements.txt
```

Backend **tái sử dụng trực tiếp** các module trong `../inference/`
(`config_loader`, `inference_engine`, `tts_generator`, `nlp_generator`, `download_female_ref`)
và đọc đường dẫn model/giọng từ `../inference/inference_config.yaml`.

## Cài đặt

Các deps inference (torch, librosa, google-genai…) đã cài theo `inference/`. Chỉ cần thêm web deps:

```bash
pip install -r webui/requirements.txt
```

## Chạy

Từ thư mục `TTS_StyleTTS2-lite-vi/`:

```bash
uvicorn webui.backend.app:app --port 8000
# hoặc
python webui/backend/app.py
```

Lần đầu sẽ **nạp model (~10–20s)** rồi in `Engine sẵn sàng`. Mở: <http://127.0.0.1:8000>

> Bước "Sinh kịch bản" gọi **Gemini** → cần `GEMINI_API_KEY` trong `Project_Final/.env`
> và mạng internet. Các bước TTS (Playground, Tổng hợp) chạy offline trên GPU.

## Luồng dùng

1. **Audiobook**: dán/tải truyện → *Sinh kịch bản* (Gemini) → sửa bảng (vai, lời, pause)
   → chọn/upload giọng nam & nữ → chỉnh tốc độ → *Tổng hợp audiobook* (có thanh tiến độ) → nghe/tải.
2. **Playground**: nhập 1 câu → chọn giọng → *Đọc thử* ngay.

## Ghi chú kỹ thuật

- Engine load **một lần** lúc startup, giữ trong VRAM (FP16, hợp 3050Ti 4GB).
- Mọi lời gọi GPU được **serialize** bằng 1 lock; audiobook chạy ở job nền (1 worker).
- Style vector được **cache** theo (file giọng, denoise, split_dur).
- Giọng upload được **health-check** (định dạng, độ dài, RMS) trước khi dùng.
