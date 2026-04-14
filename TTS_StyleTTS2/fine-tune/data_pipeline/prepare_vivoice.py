# prepare_vivoice.py
# Mục tiêu: Xử lý dataset ViVoice (HuggingFace) thành format chuẩn cho StyleTTS2 tiếng Việt.
# Đầu ra:
#   - {output_dir}/wavs/         : Các file .wav (24kHz, Mono, 16-bit PCM)
#   - {output_dir}/train_list.txt: LJSpeech format: /abs/path.wav|phoneme_string
#   - {output_dir}/val_list.txt  : LJSpeech format: /abs/path.wav|phoneme_string
#   - {output_dir}/phoneme_vocab.json : Mapping ký tự IPA -> ID (để lấy n_token)
#
# Cách chạy:
#   python prepare_vivoice.py --config configs/prepare_vivoice_config.yaml
#   python prepare_vivoice.py  # Dùng giá trị mặc định trong dataclass

import os
import sys
import json
import logging
import argparse
import random
import re
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

import yaml
import numpy as np

# =============================================================================
# CONFIGURATION — Theo phong cách RT-DETR: dataclass + from_yaml
# =============================================================================

@dataclass
class PrepareViVoiceConfig:
    """Cấu hình cho pipeline chuẩn bị dữ liệu ViVoice."""

    # --- HuggingFace Dataset ---
    hf_dataset_name: str = "capleaf/viVoice"
    hf_cache_dir: str = "D:/HUST_Project/Project_Final/data/hf_cache"
    # Tên subset/config của dataset nếu có (để "" nếu không có)
    hf_dataset_config: str = ""
    # Split để lấy data (thường là "train")
    hf_split: str = "train"

    # --- Đường dẫn output ---
    output_dir: str = "D:/HUST_Project/Project_Final/data/vivoice_processed"

    # --- Xử lý Audio ---
    target_sr: int = 24000        # Sample rate chuẩn StyleTTS2
    target_channels: int = 1      # Mono
    # Giới hạn số file xử lý (0 = không giới hạn, xử lý toàn bộ)
    max_samples: int = 0

    # --- Tỉ lệ Train/Val split ---
    val_ratio: float = 0.05       # 5% dùng làm val

    # --- Filelist output ---
    train_list_filename: str = "train_list.txt"
    val_list_filename:   str = "val_list.txt"
    vocab_filename:      str = "phoneme_vocab.json"

    # --- Xử lý Text / Phoneme ---
    # Ký tự đặc biệt được giữ lại sau khi viphoneme chuyển đổi
    # khoảng trắng " " rất quan trọng vì viphoneme dùng space phân tách âm vị
    min_phoneme_length: int = 5   # Bỏ qua các câu quá ngắn sau phonemization
    max_phoneme_length: int = 800  # Bỏ qua các câu quá dài

    # --- Reproducibility ---
    random_seed: int = 42

    # --- Logging ---
    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PrepareViVoiceConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        # Chỉ lấy các key hợp lệ, bỏ qua key lạ
        valid_keys = cls.__dataclass_fields__.keys()
        filtered = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered)


# =============================================================================
# LOGGING SETUP — Rõ ràng, có timestamp, in ra console
# =============================================================================

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("PrepareViVoice")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


# =============================================================================
# BƯỚC 1: LOAD DATASET TỪ HUGGING FACE
# =============================================================================

def load_vivoice_dataset(config: PrepareViVoiceConfig, logger: logging.Logger):
    """
    Tải dataset ViVoice từ HuggingFace về local cache.
    Trả về HuggingFace Dataset object.
    """
    logger.info("=" * 60)
    logger.info("BƯỚC 1: LOAD DATASET TỪ HUGGING FACE")
    logger.info("=" * 60)
    logger.info(f"  Dataset : {config.hf_dataset_name}")
    logger.info(f"  Cache   : {config.hf_cache_dir}")
    logger.info(f"  Split   : {config.hf_split}")

    try:
        from datasets import load_dataset, DownloadConfig
    except ImportError:
        logger.error("Thư viện 'datasets' chưa được cài. Chạy: pip install datasets")
        sys.exit(1)

    os.makedirs(config.hf_cache_dir, exist_ok=True)

    load_kwargs = dict(
        path=config.hf_dataset_name,
        split=config.hf_split,
        cache_dir=config.hf_cache_dir,
        trust_remote_code=True,
    )
    if config.hf_dataset_config:
        load_kwargs["name"] = config.hf_dataset_config

    logger.info("  Đang tải dataset (có thể mất vài phút lần đầu)...")
    dataset = load_dataset(**load_kwargs)

    total = len(dataset)
    logger.info(f"  ✅ Tải xong! Tổng số mẫu: {total:,}")

    # In thông tin cột để debug
    logger.info(f"  Các cột có sẵn: {dataset.column_names}")
    if total > 0:
        sample = dataset[0]
        for col, val in sample.items():
            if col == "audio":
                logger.info(f"    audio: sr={val.get('sampling_rate')}, "
                            f"shape={np.array(val['array']).shape}")
            else:
                preview = str(val)[:80]
                logger.info(f"    {col}: {preview}")

    return dataset


# =============================================================================
# BƯỚC 2: RESAMPLE & LƯU FILE WAV
# =============================================================================

def resample_and_save(
    audio_array: np.ndarray,
    src_sr: int,
    target_sr: int,
    save_path: str,
    logger: logging.Logger
) -> bool:
    """
    Resample audio sang target_sr, chuyển Mono, lưu 16-bit PCM WAV.
    Trả về True nếu thành công.
    """
    try:
        import torchaudio
        import torch
        import soundfile as sf
    except ImportError as e:
        logger.error(f"Thiếu thư viện audio: {e}. Cài: pip install torchaudio soundfile")
        sys.exit(1)

    # Chuyển numpy -> torch tensor, shape: [1, num_samples] (mono)
    waveform = torch.from_numpy(audio_array.astype(np.float32))

    # Nếu stereo (2D), lấy trung bình các channel để thành mono
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)      # [1, T]
    elif waveform.dim() == 2:
        waveform = waveform.mean(dim=0, keepdim=True)  # [1, T]

    # Resample nếu cần
    if src_sr != target_sr:
        resampler = torchaudio.transforms.Resample(
            orig_freq=src_sr,
            new_freq=target_sr
        )
        waveform = resampler(waveform)

    # Normalize để tránh clipping khi lưu 16-bit
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak * 0.95

    # Lưu sang 16-bit PCM WAV
    audio_np = waveform.squeeze(0).numpy()
    sf.write(save_path, audio_np, target_sr, subtype="PCM_16")
    return True


# =============================================================================
# BƯỚC 3: PHONEMIZATION — viphoneme.vi2IPA
# =============================================================================

def load_viphoneme(logger: logging.Logger):
    """
    Load thư viện viphoneme. Trả về module.
    """
    try:
        import viphoneme
        logger.info("  ✅ viphoneme loaded thành công.")
        return viphoneme
    except ImportError:
        logger.error("Thư viện 'viphoneme' chưa được cài. Chạy: pip install viphoneme")
        sys.exit(1)


def clean_text_before_phoneme(text: str) -> str:
    """
    Làm sạch text thô trước khi đưa vào viphoneme:
    - Bỏ ký tự đặc biệt rườm rà (giữ dấu câu cơ bản)
    - Chuyển số thành chữ (để viphoneme xử lý chính xác hơn)
    - Normalize khoảng trắng
    """
    # Bỏ các ký tự không cần thiết (ngoặc đơn, ngoặc kép, gạch ngang kép...)
    text = re.sub(r'["""''«»\(\)\[\]\{\}]', '', text)
    text = re.sub(r'[-–—]{2,}', ' ', text)
    text = re.sub(r'[-–—]', ' ', text)

    # Chỉ giữ lại chữ cái, dấu câu cơ bản (,.?!) và khoảng trắng
    # Chú ý: giữ lại Unicode tiếng Việt
    text = re.sub(r'[^\w\s,\.?!àáảãạăắặẳẵầấậẩẫâầấậẩẫèéẻẽẹêềếệểễìíỉĩịòóỏõọôồốộổỗơờớợởỡùúủũụưừứựửữỳýỷỹỵđÀÁẢÃẠĂẮẶẲẴẦẤẬẨẪÂÈÉẺẼẸÊỀẾỆỂỄÌÍỈĨỊÒÓỎÕỌÔỒỐỘỔỖƠỜỚỢỞỠÙÚỦŨỤƯỪỨỰỬỮỲÝỶỸỴĐ]', ' ', text)

    # Normalize khoảng trắng
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def text_to_phoneme(text: str, viphoneme_module) -> Optional[str]:
    """
    Chuyển text tiếng Việt sang chuỗi âm vị IPA dùng viphoneme.vi2IPA().
    Trả về None nếu thất bại hoặc chuỗi rỗng.
    """
    try:
        cleaned = clean_text_before_phoneme(text)
        if not cleaned:
            return None
        phoneme_str = viphoneme_module.vi2IPA(cleaned)
        if not phoneme_str or not phoneme_str.strip():
            return None
        return phoneme_str.strip()
    except Exception:
        return None


# =============================================================================
# BƯỚC 4: BUILD VOCABULARY — Đếm n_token
# =============================================================================

def build_vocabulary(phoneme_list: list, logger: logging.Logger) -> dict:
    """
    Thu thập toàn bộ ký tự IPA duy nhất từ danh sách phoneme strings.
    Trả về dictionary: {char: id} (đã thêm PAD và UNK token đặc biệt).

    Đây là hàm CỐT LÕI để xác định n_token cho StyleTTS2 config.
    """
    logger.info("=" * 60)
    logger.info("BƯỚC 4: BUILD PHONEME VOCABULARY")
    logger.info("=" * 60)

    all_chars = set()
    for phoneme_str in phoneme_list:
        if phoneme_str:
            for ch in phoneme_str:
                all_chars.add(ch)

    logger.info(f"  Tổng số ký tự IPA duy nhất (raw): {len(all_chars)}")
    logger.info(f"  Các ký tự tìm thấy: {sorted(all_chars)}")

    # Thêm các token đặc biệt (StyleTTS2 cần PAD ở index 0)
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

    # Sắp xếp để vocab ổn định (reproducible)
    sorted_chars = sorted(list(all_chars))

    # Xây dựng vocab: special tokens trước, sau đó các ký tự IPA
    vocab = {}
    for idx, token in enumerate(special_tokens):
        vocab[token] = idx

    for idx, char in enumerate(sorted_chars):
        vocab[char] = len(special_tokens) + idx

    n_token = len(vocab)

    logger.info(f"")
    logger.info(f"  📊 VOCAB BREAKDOWN:")
    logger.info(f"     Special tokens    : {len(special_tokens)}")
    logger.info(f"     IPA chars         : {len(sorted_chars)}")
    logger.info(f"  {'='*40}")
    logger.info(f"  ✅ n_token THỰC TẾ = {n_token}")
    logger.info(f"  {'='*40}")
    logger.info(f"  → Hãy dùng con số này để điền vào config_stage*.yaml")
    logger.info(f"")

    return vocab


# =============================================================================
# PIPELINE CHÍNH — Xử lý tuần tự từng mẫu
# =============================================================================

def run_pipeline(config: PrepareViVoiceConfig, logger: logging.Logger):
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║       PREPARE VIVOICE — StyleTTS2 Data Pipeline          ║")
    logger.info("╚══════════════════════════════════════════════════════════╝")
    logger.info(f"  Output dir   : {config.output_dir}")
    logger.info(f"  Target SR    : {config.target_sr} Hz")
    logger.info(f"  Val ratio    : {config.val_ratio * 100:.0f}%")
    logger.info(f"  Random seed  : {config.random_seed}")
    logger.info("")

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    # Tạo thư mục output
    wav_dir = os.path.join(config.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Bước 1: Load dataset
    # -------------------------------------------------------------------------
    dataset = load_vivoice_dataset(config, logger)

    # Giới hạn số mẫu nếu được cấu hình (dùng để test nhanh)
    if config.max_samples > 0:
        total = min(config.max_samples, len(dataset))
        logger.info(f"  ⚠️  max_samples={config.max_samples}, chỉ xử lý {total} mẫu đầu tiên.")
        dataset = dataset.select(range(total))

    # -------------------------------------------------------------------------
    # Bước 2: Load viphoneme
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 2: LOAD VIPHONEME")
    logger.info("=" * 60)
    viphoneme = load_viphoneme(logger)

    # -------------------------------------------------------------------------
    # Bước 3: Xử lý từng mẫu (Resample + Phonemize)
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 3: RESAMPLE AUDIO & PHONEMIZE TEXT")
    logger.info("=" * 60)

    # Tự động detect tên cột text trong dataset
    # ViVoice thường có cột 'transcription', 'text', hoặc 'sentence'
    text_col_candidates = ["transcription", "text", "sentence", "transcript", "label"]
    text_col = None
    for candidate in text_col_candidates:
        if candidate in dataset.column_names:
            text_col = candidate
            break
    if text_col is None:
        logger.error(f"Không tìm thấy cột text! Các cột có sẵn: {dataset.column_names}")
        logger.error("Hãy kiểm tra lại và set đúng tên cột trong config.")
        sys.exit(1)
    logger.info(f"  Dùng cột text: '{text_col}'")

    # Tự động detect tên cột audio
    audio_col = "audio" if "audio" in dataset.column_names else None
    if audio_col is None:
        logger.error(f"Không tìm thấy cột 'audio'! Các cột: {dataset.column_names}")
        sys.exit(1)

    successful_entries = []   # [(wav_abs_path, phoneme_str)]
    all_phoneme_strings = []  # Chỉ để build vocab

    n_total     = len(dataset)
    n_success   = 0
    n_skip_text = 0
    n_skip_audio = 0
    n_error     = 0

    log_every = max(1, n_total // 20)  # In tiến độ mỗi 5%

    logger.info(f"  Tổng số mẫu cần xử lý: {n_total:,}")
    logger.info(f"  Bắt đầu xử lý...")
    logger.info("")

    for idx in range(n_total):
        try:
            sample = dataset[idx]

            # --- Lấy text và phonemize ---
            raw_text = sample.get(text_col, "")
            if not raw_text or not isinstance(raw_text, str):
                n_skip_text += 1
                continue

            phoneme_str = text_to_phoneme(raw_text, viphoneme)
            if phoneme_str is None:
                n_skip_text += 1
                continue

            # Lọc độ dài phoneme
            ph_len = len(phoneme_str)
            if ph_len < config.min_phoneme_length or ph_len > config.max_phoneme_length:
                n_skip_text += 1
                continue

            # --- Lấy audio và resample ---
            audio_data = sample.get(audio_col, None)
            if audio_data is None:
                n_skip_audio += 1
                continue

            audio_array = np.array(audio_data["array"], dtype=np.float32)
            src_sr      = int(audio_data["sampling_rate"])

            # Kiểm tra audio có nội dung không (tránh lưu file rỗng)
            if len(audio_array) < src_sr * 0.5:  # Bỏ qua clip ngắn hơn 0.5 giây
                n_skip_audio += 1
                continue

            # Đặt tên file wav
            wav_filename = f"vivoice_{idx:07d}.wav"
            wav_path     = os.path.join(wav_dir, wav_filename)
            wav_abs_path = os.path.abspath(wav_path)

            # Resample & Lưu
            ok = resample_and_save(
                audio_array=audio_array,
                src_sr=src_sr,
                target_sr=config.target_sr,
                save_path=wav_path,
                logger=logger
            )

            if not ok:
                n_skip_audio += 1
                continue

            successful_entries.append((wav_abs_path, phoneme_str))
            all_phoneme_strings.append(phoneme_str)
            n_success += 1

        except Exception as e:
            n_error += 1
            if n_error <= 10:  # Chỉ in 10 lỗi đầu để tránh spam
                logger.warning(f"  Lỗi tại idx={idx}: {e}")
            continue

        # In tiến độ
        if (idx + 1) % log_every == 0 or (idx + 1) == n_total:
            pct = (idx + 1) / n_total * 100
            logger.info(
                f"  [{pct:5.1f}%] Đã xử lý {idx+1:,}/{n_total:,} | "
                f"✅ Success: {n_success:,} | "
                f"⏭️  Skip text: {n_skip_text:,} | "
                f"⏭️  Skip audio: {n_skip_audio:,} | "
                f"❌ Error: {n_error:,}"
            )

    logger.info("")
    logger.info("  📊 KẾT QUẢ XỬ LÝ:")
    logger.info(f"     Thành công : {n_success:,} mẫu")
    logger.info(f"     Bỏ qua (text) : {n_skip_text:,} mẫu")
    logger.info(f"     Bỏ qua (audio): {n_skip_audio:,} mẫu")
    logger.info(f"     Lỗi           : {n_error:,} mẫu")

    if n_success == 0:
        logger.error("Không có mẫu nào xử lý thành công! Kiểm tra lại dataset và config.")
        sys.exit(1)

    # -------------------------------------------------------------------------
    # Bước 4: Build Vocabulary
    # -------------------------------------------------------------------------
    logger.info("")
    vocab = build_vocabulary(all_phoneme_strings, logger)

    # -------------------------------------------------------------------------
    # Bước 5: Train/Val Split
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 5: TRAIN / VAL SPLIT")
    logger.info("=" * 60)

    # Shuffle trước khi split để phân phối đều
    indices = list(range(len(successful_entries)))
    random.shuffle(indices)

    n_val   = max(1, int(len(successful_entries) * config.val_ratio))
    n_train = len(successful_entries) - n_val

    train_indices = indices[n_val:]
    val_indices   = indices[:n_val]

    train_entries = [successful_entries[i] for i in train_indices]
    val_entries   = [successful_entries[i] for i in val_indices]

    logger.info(f"  Train set : {len(train_entries):,} mẫu ({(1-config.val_ratio)*100:.0f}%)")
    logger.info(f"  Val set   : {len(val_entries):,} mẫu ({config.val_ratio*100:.0f}%)")

    # -------------------------------------------------------------------------
    # Bước 6: Lưu Filelist
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 6: LƯU FILELIST & VOCAB")
    logger.info("=" * 60)

    train_list_path = os.path.join(config.output_dir, config.train_list_filename)
    val_list_path   = os.path.join(config.output_dir, config.val_list_filename)
    vocab_path      = os.path.join(config.output_dir, config.vocab_filename)

    # Ghi train_list.txt — format: /abs/path.wav|phoneme_string
    with open(train_list_path, "w", encoding="utf-8") as f:
        for wav_path, phoneme_str in train_entries:
            f.write(f"{wav_path}|{phoneme_str}\n")
    logger.info(f"  ✅ Đã lưu: {train_list_path}")

    # Ghi val_list.txt
    with open(val_list_path, "w", encoding="utf-8") as f:
        for wav_path, phoneme_str in val_entries:
            f.write(f"{wav_path}|{phoneme_str}\n")
    logger.info(f"  ✅ Đã lưu: {val_list_path}")

    # Ghi phoneme_vocab.json
    vocab_data = {
        "vocab": vocab,
        "n_token": len(vocab),
        "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
        "note": (
            "n_token này dùng để điền vào model_params.n_token trong "
            "config_stage*.yaml của StyleTTS2"
        )
    }
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab_data, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ Đã lưu: {vocab_path}")

    # -------------------------------------------------------------------------
    # TỔNG KẾT CUỐI — In rõ ràng để dễ copy sang config
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("╔══════════════════════════════════════════════════════════╗")
    logger.info("║                    TỔNG KẾT                              ║")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║  Wavs đã lưu  : {wav_dir}")
    logger.info(f"║  Train list   : {train_list_path}")
    logger.info(f"║  Val list     : {val_list_path}")
    logger.info(f"║  Vocab file   : {vocab_path}")
    logger.info("╠══════════════════════════════════════════════════════════╣")
    logger.info(f"║")
    logger.info(f"║  ⭐ THÔNG SỐ ĐỂ ĐIỀN VÀO config_stage*.yaml :")
    logger.info(f"║     model_params:")
    logger.info(f"║       n_token: {len(vocab)}")
    logger.info(f"║")
    logger.info(f"║  → Copy con số {len(vocab)} này vào cả 3 file config!")
    logger.info("╚══════════════════════════════════════════════════════════╝")


# =============================================================================
# ENTRY POINT
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset ViVoice cho StyleTTS2 tiếng Việt.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Đường dẫn tới file YAML config.\n"
             "Nếu không truyền, dùng giá trị mặc định trong dataclass.\n"
             "Ví dụ: --config configs/prepare_vivoice_config.yaml"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Override số mẫu tối đa (dùng để test nhanh, ví dụ: --max_samples 100)"
    )
    args = parser.parse_args()

    # Load config từ YAML hoặc dùng default
    if args.config and os.path.exists(args.config):
        config = PrepareViVoiceConfig.from_yaml(args.config)
        print(f"[INFO] Loaded config từ: {args.config}")
    else:
        config = PrepareViVoiceConfig()
        if args.config:
            print(f"[WARN] Không tìm thấy config file: {args.config}. Dùng giá trị mặc định.")
        else:
            print(f"[INFO] Không có --config. Dùng giá trị mặc định trong PrepareViVoiceConfig.")

    # Override từ CLI nếu có
    if args.max_samples is not None:
        config.max_samples = args.max_samples
        print(f"[INFO] Override max_samples = {config.max_samples}")

    # Setup logging
    logger = setup_logging(config.log_level)

    # Chạy pipeline
    run_pipeline(config, logger)


if __name__ == "__main__":
    main()