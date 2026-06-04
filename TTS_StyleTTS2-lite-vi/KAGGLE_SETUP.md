# KAGGLE_SETUP.md — Hướng dẫn setup Kaggle Free để Fine-tune StyleTTS2-lite-vi

> **Mục đích**: Hướng dẫn đầy đủ từ A → Z cách upload data Ngạn lên Kaggle, tạo notebook training, quản lý quota 30h/tuần và resume training qua nhiều phiên. File này đọc 1 lần là đủ.

---

## Mục lục

1. [Tổng quan & Pre-flight Checklist](#1-tổng-quan--pre-flight-checklist)
2. [Hạn chế của Kaggle Free Tier — cần biết trước](#2-hạn-chế-của-kaggle-free-tier--cần-biết-trước)
3. [Chuẩn bị data trên máy local](#3-chuẩn-bị-data-trên-máy-local)
4. [Tạo Kaggle Dataset (làm 1 lần)](#4-tạo-kaggle-dataset-làm-1-lần)
5. [Tạo & cấu hình Kaggle Notebook](#5-tạo--cấu-hình-kaggle-notebook)
6. [Storage strategy — phải hiểu để không mất checkpoint](#6-storage-strategy--phải-hiểu-để-không-mất-checkpoint)
7. [Save & Resume training qua nhiều phiên](#7-save--resume-training-qua-nhiều-phiên)
8. [Download checkpoint về local](#8-download-checkpoint-về-local)
9. [Troubleshooting](#9-troubleshooting)
10. [Quota planning & timeline 6 tuần](#10-quota-planning--timeline-6-tuần)

---

## 1. Tổng quan & Pre-flight Checklist

### Workflow tổng thể

```
[MÁY LOCAL]
  ├─ Data Ngạn (~1-2 GB wavs + filelists)
  └─ Zip → upload
            ↓
[KAGGLE DATASET]                                    ← bước này 1 LẦN
  └─ ngan-data-lite-vi (Private)
            ↓ attach
[KAGGLE NOTEBOOK]                                   ← chạy nhiều phiên
  ├─ Clone source code StyleTTS2-lite
  ├─ Tải pretrained checkpoint
  ├─ Train ~30-50 epoch (chia 3-4 phiên × 8h)
  └─ Save checkpoint → output zip
            ↓
[MÁY LOCAL]
  └─ Download checkpoint best.pth → inference local
```

### Pre-flight Checklist (cần có trước khi bắt đầu)

| Mục | Tình trạng |
|-----|-----------|
| Kaggle account (free) | ☐ |
| Đã xác minh số điện thoại (cần thiết để dùng GPU) | ☐ |
| Đã chạy xong step0 → step1b → A2 → A3.2 | ☐ |
| File `ngan_train_lite.txt` & `ngan_val_lite.txt` đã có | ☐ |
| Folder `wavs/` chứa các file `.wav` 24kHz mono | ☐ |
| Đã verify zero-shot test (A3.2) cho kết quả OK | ☐ |
| Có connection internet ổn định (upload ~1-2 GB) | ☐ |

> ⚠️ **Xác minh số điện thoại**: Kaggle yêu cầu verify số điện thoại để dùng GPU/TPU/Internet. Vào **Account → Settings → Phone Verification**. KHÔNG verify thì GPU sẽ không bật được dù chọn trong Accelerator.

---

## 2. Hạn chế của Kaggle Free Tier — cần biết TRƯỚC

| Resource | Hạn chế Free | Implication |
|----------|--------------|-------------|
| **GPU quota** | 30 giờ/tuần (reset thứ 7 hàng tuần) | Đủ cho ~3-4 phiên training × 8h |
| **Session timeout** | 9-12h tự động kill (idle hoặc max) | Phải save checkpoint định kỳ + resume |
| **GPU options** | T4 ×2 (15GB ×2), ~~P100 16GB~~, TPU v5e-8 | **CHỈ DÙNG T4 ×2** (P100 đã bị PyTorch 2.10 drop support) |
| **Dataset size** | 100 GB/dataset (private) | Data Ngạn ~1-2 GB → thoải mái |
| **Dataset count** | Unlimited datasets | OK |
| **Notebook disk** | ~73 GB ephemeral `/kaggle/working` | Đủ cho checkpoint + log |
| **RAM** | ~30 GB | Đủ |
| **Internet** | BẬT trong Settings (default OFF) | Cần để pip install + clone HuggingFace |
| **Persistent files** | Chỉ `/kaggle/working/` khi click "Save Version" | KHÔNG save = mất hết khi session kill |

> 🔑 **Quy tắc vàng**: Mọi checkpoint training PHẢI lưu vào `/kaggle/working/`. Mọi data input PHẢI nằm trong Kaggle Dataset (không upload trực tiếp vào notebook).

---

## 3. Chuẩn bị data trên máy local

### 3.1. Cấu trúc folder cần upload

Tạo folder mới trên máy local, ví dụ `D:\Kaggle_Upload\ngan-data-lite-vi\` với cấu trúc:

```
ngan-data-lite-vi/
├── wavs/
│   ├── ngan_00001.wav
│   ├── ngan_00002.wav
│   ├── ...
│   └── ngan_02165.wav            (tổng ~2155 file sau cleanup)
├── ngan_train_lite.txt           (filelist train từ A2)
└── ngan_val_lite.txt             (filelist val từ A2)
```

**Lưu ý quan trọng về path trong filelist**:
- File `ngan_train_lite.txt` chứa các dòng `wavs/ngan_00001.wav|<phoneme>` (forward slash)
- Khi training trên Kaggle, ta sẽ set `root_path: /kaggle/input/ngan-data-lite-vi/` → file_path resolve thành `/kaggle/input/ngan-data-lite-vi/wavs/ngan_00001.wav` ✓
- A2 đã xử lý normalize backslash → forward slash. Nếu chưa, hãy verify bằng: mở filelist trong text editor, search `\` → phải KHÔNG còn.

### 3.2. Lệnh copy data từ pipeline cũ

Trên Windows PowerShell:

```powershell
# Tạo folder upload
$UPLOAD = "D:\Kaggle_Upload\ngan-data-lite-vi"
New-Item -ItemType Directory -Force -Path "$UPLOAD\wavs"

# Copy wavs từ pipeline cũ
$SRC = "D:\Documents\HUST\HUST_Project\Project_Final\data\StyleTTS2_preprocess\output_dataset\wavs"
Copy-Item "$SRC\*.wav" -Destination "$UPLOAD\wavs\"

# Copy filelists từ pipeline lite-vi
$LITE = "D:\Documents\HUST\HUST_Project\Project_Final\TTS_StyleTTS2-lite-vi\output"
Copy-Item "$LITE\ngan_train_lite.txt" -Destination "$UPLOAD\"
Copy-Item "$LITE\ngan_val_lite.txt" -Destination "$UPLOAD\"

# Verify
Get-ChildItem $UPLOAD\wavs | Measure-Object | Select-Object Count
Get-Content "$UPLOAD\ngan_train_lite.txt" | Measure-Object -Line
```

**Expected output**: ~2155 wav files, ~2055 dòng train, ~110 dòng val.

### 3.3. Kích thước cuối cùng

```powershell
# Check tổng kích thước
"{0:N2} MB" -f ((Get-ChildItem $UPLOAD -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB)
```

Kỳ vọng: **800 MB - 1.5 GB**. Nếu lớn hơn 5 GB → xem lại pipeline preprocess (max_duration_s).

### 3.4. KHÔNG cần zip thủ công

Kaggle UI sẽ zip TỰ ĐỘNG khi upload. Bạn chỉ cần drag whole folder vào browser. Đừng tạo `.zip` thủ công vì Kaggle sẽ extract lại (mất thời gian gấp đôi).

---

## 4. Tạo Kaggle Dataset (làm 1 lần)

### 4.1. Vào trang Datasets

1. Đăng nhập https://www.kaggle.com
2. Click "Datasets" ở sidebar trái → click **"+ New Dataset"** (nút màu xanh)

### 4.2. Upload files

1. Trong popup mở ra, drag **CẢ FOLDER** `ngan-data-lite-vi/` vào upload area
   - Browser sẽ hiển thị danh sách file (~2155 files)
   - Kaggle giữ nguyên cấu trúc folder con
2. **Đợi upload hoàn tất** — với 1-2 GB và mạng 50 Mbps, khoảng 10-20 phút
   - KHÔNG đóng tab browser trong lúc upload
   - Nếu mạng yếu, có thể đêm khuya upload tốt hơn

### 4.3. Đặt thông tin dataset

Sau khi upload xong:

| Field | Giá trị | Lý do |
|-------|---------|-------|
| **Title** | `ngan-data-lite-vi` | Tên ngắn gọn, không space |
| **Subtitle** | `Vietnamese audiobook narrator dataset for StyleTTS2-lite-vi fine-tuning` | (optional) |
| **License** | Other (Specified in description) | Data của bạn, chưa public-domain |
| **Visibility** | 🔒 **Private** | QUAN TRỌNG — giọng Ngạn không nên public |

### 4.4. Create & verify

1. Click **"Create"** (góc dưới phải)
2. Đợi Kaggle process (~2-5 phút)
3. Sau khi xong, page sẽ hiển thị dataset với 3 tab: Data, Code, Discussion
4. **Verify cấu trúc**: click tab "Data" → bạn phải thấy:
   ```
   ngan-data-lite-vi/
   ├── wavs/  (folder, có hơn 2000 files)
   ├── ngan_train_lite.txt
   └── ngan_val_lite.txt
   ```
5. **Copy đường dẫn dataset slug** (xem URL): ví dụ `vieanh88/ngan-data-lite-vi` — sẽ cần ở bước attach.

> 💡 **Tip**: Lần sau muốn update data (vd: thêm sample), KHÔNG cần tạo dataset mới. Vào dataset đã có → "New Version" → upload file mới → Kaggle giữ lịch sử versions.

---

## 5. Tạo & cấu hình Kaggle Notebook

### 5.1. Tạo notebook mới

1. Sidebar trái → click "Code" → **"+ New Notebook"** (nút xanh)
2. Notebook editor mở ra với 1 cell mặc định

### 5.2. Cấu hình Settings (sidebar phải)

Click icon ⚙️ "Settings" sidebar phải, set:

| Setting | Giá trị | Lý do |
|---------|---------|-------|
| **Language** | Python | (default) |
| **Environment** | Pin to original environment (cập nhật mới nhất) | Để có PyTorch 2.10+ |
| **Accelerator** | **GPU T4 ×2** | T4 ×2 = 30GB tổng VRAM, đủ batch lớn |
| **Persistence** | Files only | Sẽ save `/kaggle/working` khi "Save Version" |
| **Internet** | **ON** | Cần để pip install + clone HuggingFace |

> ⚠️ **TUYỆT ĐỐI KHÔNG chọn P100**. PyTorch 2.10+ đã drop sm_60 → P100 sẽ fail.
> ⚠️ **TUYỆT ĐỐI KHÔNG chọn TPU** (trừ khi bạn muốn rewrite training loop với JAX/XLA — không cần).

### 5.3. Attach dataset

Sidebar phải → mục "Input":
1. Click **"+ Add Input"**
2. Tab "Datasets" → search `ngan-data-lite-vi` (tên dataset của bạn)
3. Click "+" bên cạnh dataset → dataset được mount vào `/kaggle/input/ngan-data-lite-vi/`
4. **Verify**: trong cell đầu tiên gõ và Run:
   ```python
   !ls -lah /kaggle/input/ngan-data-lite-vi/
   !head -3 /kaggle/input/ngan-data-lite-vi/ngan_train_lite.txt
   ```
   Phải thấy folder `wavs/` và 2 file `.txt`. Nếu KHÔNG thấy → recheck step 5.3.

### 5.4. Đặt tên notebook

Click trên cùng (chỗ "Untitled") → đổi thành `kaggle_finetune_ngan_lite_vi` (hoặc tên tùy ý). Giúp bạn dễ tìm trong "Your Work" sau này.

### 5.5. (Optional) Bật GPU monitoring widget

Trên thanh top, có 5 indicator HDD/CPU/RAM/GPU/Power. Click vào GPU → hiển thị real-time VRAM usage. **Để mở trong khi train** để biết OOM ngay khi xảy ra.

---

## 6. Storage strategy — phải hiểu để không mất checkpoint

Kaggle có 3 loại storage, **mỗi loại có hành vi rất khác nhau**:

### `/kaggle/input/` — READ-ONLY, persistent

- Mount tự động khi attach Dataset
- KHÔNG thể ghi vào (sẽ raise PermissionError)
- Tồn tại mãi (dataset là static)
- Dùng cho: data training, pretrained checkpoint từ HF

### `/kaggle/working/` — READ-WRITE, persistent CÓ ĐIỀU KIỆN

- **Persistent CHỈ KHI** bạn click "Save Version" → "Save & Run All" → đợi notebook chạy xong + commit
- KHÔNG save = **MẤT TẤT CẢ** khi session timeout (9-12h) hoặc bạn đóng tab
- Disk quota: **20 GB persistent** (khi commit). Vượt → commit fail.
- Dùng cho: **checkpoint training** (`best.pth`, `current.pth`), output logs

> 🚨 **Cảnh báo nghiêm trọng**: Khi training kéo dài (vd: 8 giờ), TUYỆT ĐỐI không đóng tab browser. Nếu cần đóng → phải "Save Version" trước. Nếu không → mất hết checkpoint.

### `/tmp/`, `/root/`, vv. — EPHEMERAL

- Mất khi session kill
- Dùng cho: cache pip, temp files

### Disk usage tracking

Thêm vào notebook (cell debug):
```python
!df -h /kaggle/working /kaggle/input /tmp
```

---

## 7. Save & Resume training qua nhiều phiên

Vì 1 phiên Kaggle ≤ 9-12h và bạn cần 30-50 epoch (~24-40h training), bạn phải chia **3-4 phiên** và resume.

### Strategy

```
Phiên 1: epoch 0 → 12  → Save best.pth → "Save Version"
Phiên 2: load best.pth → epoch 12 → 25  → Save → "Save Version"
Phiên 3: load best.pth → epoch 25 → 40  → Save → "Save Version"
Phiên 4: load best.pth → epoch 40 → 50  → Save final → Download
```

### Cơ chế resume — sẽ implement trong file B2

Trong config training, ta dùng:
```yaml
pretrained_model: "<path_to_previous_checkpoint>"
load_only_params: false   # quan trọng: load CẢ optimizer state + epoch_count
```

- Phiên đầu: `pretrained_model: ./Models/Finetune/base_model.pth` (lite-vi pretrained)
- Phiên 2+: `pretrained_model: ./Models/Finetune/best.pth` (từ phiên trước, đã download lại từ "Output" của notebook commit cũ)

> 💡 **Tip quan trọng**: Mỗi lần "Save Version", checkpoint `best.pth` từ `/kaggle/working/Models/` sẽ có trong "Output" của notebook đó. Click vào → download về local → upload lên Kaggle Dataset version mới → attach vào notebook tiếp theo.

### Cách "Save Version" đúng

1. Trước khi sắp hết session (vd: 7-8 giờ chạy), trong cell cuối của notebook, RUN:
   ```python
   !ls -lah /kaggle/working/Models/
   ```
   Verify `best.pth` đã có.
2. Click nút **"Save Version"** trên cùng (góc phải)
3. Popup chọn:
   - **Save & Run All (Commit)**: chạy LẠI từ đầu notebook (NÊN dùng nếu code clean & nhanh restart). Nhưng nếu đang training dở → SẼ MẤT progress.
   - **Quick Save**: lưu CURRENT state (KHÔNG chạy lại) — **CHỌN CÁI NÀY** khi đang training.
4. Đợi save xong (~2-5 phút).
5. Sau khi save, vào tab **"Output"** của Version vừa tạo → thấy file `best.pth` → click download về local.

---

## 8. Download checkpoint về local

### Cách 1: Qua Notebook Output (recommend)

1. Sau "Save Version" → vào Version → tab "Output"
2. Click vào file `best.pth` (hoặc `current_model.pth`)
3. Click "Download" (góc phải)
4. File ~570MB sẽ download về `Downloads/` máy local

### Cách 2: Tạo Kaggle Dataset từ output

Nếu cần version control checkpoint:
1. Trong notebook đã commit, scroll xuống "Output" → click "New Dataset" 
2. Đặt tên `ngan-checkpoint-v1` (hoặc v2, v3 cho các phiên sau)
3. Dataset chứa toàn bộ `/kaggle/working/` của Version đó → có thể attach vào notebook khác để resume

### Cách 3: Public URL (KHÔNG dùng cho Private project)

Notebook output public sẽ có URL trực tiếp. **Bỏ qua vì giọng Ngạn nên Private.**

---

## 9. Troubleshooting

### "GPU not detected" sau khi setup

**Triệu chứng**: `torch.cuda.is_available()` trả về `False`.

**Nguyên nhân & fix**:
- Quên BẬT Accelerator → vào Settings, chọn GPU T4 ×2 → Save Session
- Quên verify phone → Account → Settings → Phone Verification

### "CUDA error: no kernel image available"

**Nguyên nhân**: GPU là P100 nhưng PyTorch không support nữa.

**Fix**: Settings → Accelerator → đổi sang **GPU T4 ×2** → Save Session.

### "Disk quota exceeded" khi save version

**Nguyên nhân**: `/kaggle/working/` > 20 GB.

**Fix**: Trước khi save, clean cache:
```python
!rm -rf /kaggle/working/StyleTTS2-lite-vi/.git  # ~100MB
!rm -rf /kaggle/working/Models/checkpoint_step_*.pth  # giữ chỉ best.pth
!du -sh /kaggle/working/
```

### "RuntimeError: CUDA out of memory" giữa training

**Fix sequence (từ nhẹ → nặng)**:
1. Giảm `batch_size` từ 2 → 1
2. Giảm `max_len` từ 310 → 250 → 200
3. Bật mixed precision (sẽ có flag trong B2 config)
4. Restart kernel (Run → Restart) — free fragment memory

### Internet bị off giữa chừng (Kaggle disconnect)

**Triệu chứng**: pip install bị fail giữa chừng.

**Fix**: Settings → Internet → toggle OFF rồi ON → Save Session → retry.

### Notebook bị kill sau 9 giờ dù chưa idle

**Đây là behavior NORMAL của Kaggle Free** — session limit cứng. Phải Save Version định kỳ.

**Workaround**: setup checkpoint save MỖI 1 EPOCH (sẽ có trong B2). Nếu kill → chỉ mất tối đa 1 epoch.

### Sau khi resume, loss tăng đột ngột

**Nguyên nhân**: `load_only_params: true` được set nhầm → optimizer state bị reset.

**Fix**: Trong config, set `load_only_params: false` để load CẢ Adam momentum.

---

## 10. Quota planning & timeline 6 tuần

### Phân bổ 30h GPU/tuần

| Tuần | Hoạt động | GPU hours dự kiến |
|------|-----------|-------------------|
| 1 | Setup + chạy zero-shot A3.2 + dry-run 1 epoch | 5h |
| 2 | Phiên 1 training (epoch 0-12) | 10h |
| 3 | Phiên 2 training (epoch 12-25) | 10h |
| 4 | Phiên 3 training (epoch 25-40) | 10h |
| 5 | Phiên 4 training (epoch 40-50) + first inference test | 10h |
| 6 | Polish, final inference, ghi báo cáo, demo | 5h |
| **Tổng** | | **50h** (qua 6 tuần × ~30h quota = 180h, dư xa) |

→ Bạn có **dư khoảng 130h** dự phòng cho debug/retry. Rất an toàn.

### Mẹo tiết kiệm quota

1. **Phát triển code trên CPU first**: notebook ở chế độ `Accelerator: None` để debug syntax, sau đó mới bật GPU.
2. **Không Run All khi chỉ sửa 1 cell**: dùng "Run selected cell" để tiết kiệm.
3. **Tắt kernel khi không dùng**: trên Kaggle, vào "Your Work" → bấm "Stop session" nếu còn active mà không cần.
4. **Đặt timer**: dùng phone timer 8h khi training để Save Version đúng lúc.
5. **Test trên subset nhỏ trước**: dry-run 1 epoch với 100 samples để verify code chạy được trước khi full train.

---

## Tổng kết — Action items

Sau khi đọc xong file này, bạn cần làm theo thứ tự:

1. ☐ Verify Pre-flight Checklist (mục 1)
2. ☐ Chạy script copy data (mục 3.2) → có folder `D:\Kaggle_Upload\ngan-data-lite-vi\`
3. ☐ Tạo Kaggle Dataset (mục 4) → upload + verify
4. ☐ Verify phone trên Kaggle account
5. ☐ (Sẽ làm sau khi tôi gửi B2) Tạo notebook training, attach dataset, chạy

**Khi xong các bước 1-4, báo tôi để chuyển sang B2** — `kaggle_finetune_ngan_lite_vi.ipynb`.

---

## Câu hỏi thường gặp

**Q: Tôi có thể train trên Colab Free thay vì Kaggle không?**  
A: Có nhưng Colab Free chỉ cho T4 single (15GB) và session ngắn hơn (~4-6h). Kaggle Free tốt hơn vì T4 ×2 (30GB) và 9h/session.

**Q: Nếu vượt quota 30h/tuần thì sao?**  
A: GPU bị disable đến đầu tuần sau (thứ 7 reset). Vẫn dùng được CPU. Với plan 50h tổng cho 6 tuần (~8h/tuần), bạn KHÔNG bao giờ chạm quota.

**Q: Có cần Kaggle Pro ($) không?**  
A: KHÔNG. Free tier đủ thoải mái cho project này.

**Q: Multi-GPU training có nhanh hơn không trên T4 ×2?**  
A: Có nhưng phức tạp setup. **Recommend dùng 1 GPU trước** (đặt `device='cuda:0'`), nếu thiếu thời gian thì mới setup DataParallel sau. B2 sẽ default single-GPU.

---

*File này phiên bản 1.0 — viết cho project Vietnamese Ghost Story Narrator, fine-tune StyleTTS2-lite-vi*