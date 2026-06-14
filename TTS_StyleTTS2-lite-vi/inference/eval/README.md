# Đánh giá thực nghiệm (Chương 5): MOS • SMOS • CER • RTF

Quy trình tái lập đầy đủ. Chạy mọi lệnh từ thư mục `TTS_StyleTTS2-lite-vi/`.

## 0. Cài thêm phụ thuộc (cho CER + SECS)
```bash
pip install transformers    # CER (PhoWhisper/Whisper) + SECS (WavLM) đều dùng transformers
```
RTF và render dùng đúng môi trường inference hiện có (torch, soundfile, espeak…).
SECS không cần cài thêm gì ngoài transformers (mô hình WavLM tải tự động).

## 1. Tập câu test
`testset.json` — **cố định**, không sửa sau khi đã chạy để số liệu tái lập được:
- `in_domain` (5 câu): có bản thu giọng thật → mốc trên MOS/SMOS + sàn CER.
- `out_domain` (15 câu): kinh dị / ngụ ngôn / lịch sử → CER theo thể loại.

## 2. Các bước

| Bước | Lệnh | Kết quả |
|---|---|---|
| Render audio 3 điều kiện + bộ mẫu Form | `python inference/evaluate_demo.py render` | `output/eval/audio/`, `output/eval/mos_samples/`, `output/eval/mos_manifest.csv` |
| Đo hiệu năng RTF | `python inference/evaluate_demo.py rtf` | `output/eval/rtf_results.csv`, `rtf_summary.json` |
| Tính CER (sau render) | `python inference/evaluate_demo.py cer` | `output/eval/cer_results.csv`, `cer_summary.json` |
| Tính SECS — tương đồng giọng khách quan (sau render) | `python inference/evaluate_demo.py secs` | `output/eval/secs_results.csv`, `secs_summary.json` |
| (Hoặc tất cả) | `python inference/evaluate_demo.py all` | — |

Điều kiện so sánh: `final` (sau Giai đoạn 3) · `pretrained` (mô hình nền trước fine-tune, mặc định `kaggle_models/base_model_120k_vi.pth`) · `groundtruth` (bản thu thật).

**Kết quả khách quan đã đo (RTX 3050 Ti, FP16):** CER (PhoWhisper) `final` 1,28% ≈ sàn ASR 1,37% < `pretrained` 2,92%; SECS `final` 0,977 ≈ trần 0,977 > `pretrained` 0,948; RTF FP16 ≈ 0,053.

## 3. Khảo sát MOS/SMOS
Xem **`MOS_FORM_GUIDE.md`**: dựng Google Form từ `mos_samples/`, thu ≥15 người nghe,
chuẩn hoá phản hồi về long-format rồi:
```bash
python inference/analyze_mos.py --responses responses_long.csv
# -> output/eval/mos_results.csv  (MOS/SMOS ± CI95 theo điều kiện)
```

## 4. Điền báo cáo
Số liệu từ `mos_results.csv`, `cer_summary.json`, `rtf_summary.json` điền vào các
bảng đã đánh dấu `% TODO` trong `documents/report_datn/Chuong/5_Thuc_nghiem.tex`
(Bảng `tab:mos-smos`, `tab:cer-condition`, `tab:cer-genre`, `tab:rtf`).

## Ghi chú quan trọng
- `mos_manifest.csv` là **đáp án** (sample → điều kiện thật) — **không** đưa người nghe.
- CER trên `groundtruth` = **sàn ASR**, không phải lỗi TTS.
- RTF đo bằng warm-up + `cuda.synchronize` + trung vị; **không** tính độ trễ Gemini.
- Trong văn bản báo cáo chỉ gọi "giọng mục tiêu / hệ thống đề xuất / mô hình nền".
