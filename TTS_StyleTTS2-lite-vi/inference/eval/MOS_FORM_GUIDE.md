# Hướng dẫn đánh giá chủ quan: MOS (độ tự nhiên) + SMOS (độ tương đồng giọng)

Tài liệu này hướng dẫn dựng Google Form, thu thập và xử lý điểm cho **Mục 5.2** của báo cáo.
Bộ mẫu nghe được sinh tự động bởi `python inference/evaluate_demo.py render`.

---

## 1. Đầu vào đã có sẵn sau bước `render`

| File / thư mục | Vai trò |
|---|---|
| `output/eval/mos_samples/sample_XXX.wav` | Các đoạn audio **đã ẩn danh + trộn ngẫu nhiên** để đưa người nghe. |
| `output/eval/mos_samples/SMOS_reference.wav` | Đoạn **giọng mục tiêu** dùng làm tham chiếu cho SMOS. |
| `output/eval/mos_manifest.csv` | **ĐÁP ÁN** (sample → điều kiện thật). **GIỮ RIÊNG, KHÔNG đưa người nghe.** |

Mỗi `sample_XXX.wav` thuộc một trong 3 điều kiện (người nghe **không** biết):
- `groundtruth` — bản ghi giọng người thật (mốc trên).
- `final` — mô hình sau Giai đoạn 3 (hệ thống đề xuất).
- `pretrained` — mô hình **trước** khi fine-tune giọng mục tiêu (mốc dưới).

Cột `smos_eligible = yes` đánh dấu các mẫu **in-domain** (có cả 3 điều kiện cùng một câu) → dùng cho **cả MOS lẫn SMOS**. Mẫu `no` (đa thể loại) chỉ dùng cho **MOS độ tự nhiên**.

---

## 2. Thang điểm (mỏ neo rõ ràng — đưa vào đầu Form)

### 2.1. MOS — Độ tự nhiên (Naturalness), thang 1–5
Nghe đoạn audio và chấm mức độ **tự nhiên, giống người thật nói**:

| Điểm | Mô tả |
|---|---|
| 5 | Rất tự nhiên — không phân biệt được với giọng người thật. |
| 4 | Tự nhiên — nghe thoải mái, lỗi rất nhỏ. |
| 3 | Chấp nhận được — hơi máy móc nhưng vẫn dễ nghe. |
| 2 | Kém tự nhiên — méo tiếng/ngắt nhịp lạ, khó nghe. |
| 1 | Rất tệ — robotic, khó hiểu. |

### 2.2. SMOS — Độ tương đồng giọng (Speaker Similarity), thang 1–5
*(Chỉ cho các mẫu được đánh dấu)* Nghe **đoạn tham chiếu** `SMOS_reference.wav`, rồi nghe mẫu và chấm **mức độ giống nhau về CHẤT GIỌNG** (bỏ qua nội dung):

| Điểm | Mô tả |
|---|---|
| 5 | Chắc chắn cùng một người. |
| 4 | Rất giống — gần như cùng người. |
| 3 | Hơi giống — cùng kiểu giọng. |
| 2 | Khác — chỉ giống mơ hồ. |
| 1 | Hoàn toàn khác người. |

> **Lưu ý độ rõ chữ (intelligibility):** không cần chấm bằng tay — đã đo khách quan bằng **CER** ở Mục 5.3. Nếu muốn, có thể thêm 1 câu hỏi phụ "độ rõ chữ 1–5" nhưng không bắt buộc.
>
> **SMOS đã có bản đo khách quan song song:** điểm **SECS** (cosine giọng, `evaluate_demo.py secs`) là phiên bản tự động của SMOS. Hai con số bổ trợ nhau trong báo cáo: SMOS = cảm nhận người, SECS = đo máy. Không thay thế việc thu SMOS chủ quan.

---

## 3. Dựng Google Form

1. **Tải audio lên Google Drive** (1 thư mục, đặt quyền *Bất kỳ ai có liên kết — Người xem*).
2. **Phần mở đầu**: nêu mục đích, yêu cầu **đeo tai nghe, nghe nơi yên tĩnh**, dán bảng thang điểm ở Mục 2. Hỏi vài thông tin nền (độ tuổi, có làm trong lĩnh vực âm thanh/AI không) để mô tả nhóm người nghe trong báo cáo.
3. **Mỗi mẫu = 1 câu hỏi**:
   - Nhúng audio (link Drive) hoặc câu hỏi dạng *Linear scale 1–5*.
   - Tiêu đề: "Mẫu sample_037 — Độ tự nhiên (1–5)".
   - Với mẫu `smos_eligible=yes`: thêm câu thứ hai "Mẫu sample_037 — Độ tương đồng với giọng tham chiếu (1–5)".
4. **Thứ tự**: bộ mẫu ĐÃ được script trộn ngẫu nhiên (seed cố định) → cứ giữ nguyên thứ tự `sample_001 → sample_NNN`. Không nhóm theo điều kiện để tránh lộ.
5. **Câu kiểm tra chú ý (attention check)**: chèn 1–2 câu hiển nhiên (vd một mẫu `groundtruth` rõ ràng) và loại bỏ người nghe chấm phi lý (cho điểm 1 cho giọng người thật).
6. **Quy mô**: mục tiêu **≥ 15 người nghe**.

### 3.1. Quản lý độ dài form (QUAN TRỌNG)
Bộ mẫu hiện có **45 clip** (15 in-domain × 3 điều kiện + 15 out-domain × 2 điều kiện).
Nếu bắt mỗi người chấm cả 45 (cộng 15 mục SMOS) thì ~20–25 phút → dễ mệt, giảm chất
lượng. Hai cách xử lý (chọn 1):
- **Cách A (gọn, khuyến nghị — ĐÃ tạo sẵn):** dùng **`output/eval/mos_manifest_30.csv`**
  (30 clip = 15 in-domain đủ 3 điều kiện + 15 out-domain `final`; 15 clip có SMOS).
  Chỉ đưa đúng 30 file `sample_XXX.wav` liệt kê trong file này vào form. Lý do: so
  sánh `final` vs `pretrained` đã được CER và SECS đo khách quan; MOS out-domain chỉ
  cần định vị độ tự nhiên của hệ thống đề xuất. Khi phân tích, truyền
  `--manifest output/eval/mos_manifest_30.csv` cho `analyze_mos.py`.
- **Cách B (đầy đủ):** giữ cả 45 clip nhưng **chia 2 phiên bản form**, mỗi người một
  nửa, rồi nhân đôi số người nghe để mỗi clip vẫn đủ lượt.

### 3.2. (Tuỳ chọn) CMOS — so sánh trực tiếp
Nếu sau này thêm một hệ thống đường cơ sở (mô hình TTS tiếng Việt khác), có thể thay
MOS tuyệt đối bằng **CMOS**: phát cặp A/B cùng câu, hỏi "B tự nhiên hơn A bao nhiêu"
trên thang $-3..+3$. CMOS nhạy hơn MOS khi so hai hệ tương đương, nhưng tốn thêm một
vòng khảo sát — chỉ làm khi thực sự có hệ đối chứng.

---

## 4. Thu thập & xử lý kết quả

### 4.1. Chuẩn hoá phản hồi về "long format"
Tải phản hồi Form về CSV. Cần đưa về bảng dài với các cột:

```
rater_id, sample_file, mos, smos
r01, sample_001.wav, 4, 3
r01, sample_002.wav, 5,        <- để trống smos nếu mẫu không eligible
...
```

(Google Form xuất "wide format" — mỗi mẫu một cột. Dùng `pandas.melt` hoặc thao tác tay trong Sheets để xoay về dạng trên. Tên cột cần chứa `sample_XXX` để khớp manifest.)

### 4.2. Tính MOS/SMOS trung bình + khoảng tin cậy 95%
Chạy script kèm theo:

```bash
python inference/analyze_mos.py --responses path/to/responses_long.csv
```

Script join với `output/eval/mos_manifest.csv` (theo `sample_file`), rồi tính cho **từng điều kiện** (`groundtruth` / `final` / `pretrained`):
- MOS trung bình, độ lệch chuẩn, **khoảng tin cậy 95%** = `mean ± 1.96 · std / √N`.
- SMOS tương tự (chỉ trên mẫu eligible, so `final` vs `groundtruth`).
- Xuất bảng sẵn để dán vào báo cáo.

### 4.3. Diễn giải đưa vào báo cáo
- `final` nằm **giữa** `pretrained` (dưới) và `groundtruth` (trên) → chứng minh fine-tune Giai đoạn 3 có hiệu quả.
- Báo cáo dạng **trung bình ± CI 95%**, kèm **N người nghe** và mô tả nhóm.
- SMOS cao của `final` so với `pretrained` → mô hình đã học được **chất giọng mục tiêu**.

> **Bảo mật narrative báo cáo:** trong văn bản chỉ gọi "giọng mục tiêu", "hệ thống đề xuất", "mô hình nền". Không nêu tên người thật hay tên thư viện gốc.
