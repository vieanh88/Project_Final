# prepare_vivoice.py  (v2 — fixed)
# Pipeline xử lý dataset viVoice (HuggingFace) → chuẩn StyleTTS2 tiếng Việt
# Thay đổi so với v1:
#   ✅ Bỏ trust_remote_code (deprecated, gây lỗi ngay)
#   ✅ Đọc HF_TOKEN từ biến môi trường (gated dataset bắt buộc)
#   ✅ Dùng cột 'channel' làm speaker_id → multispeaker: true từ đầu
#   ✅ Filelist format 3 cột: wav_path|phoneme|speaker_id
#   ✅ Lưu channel_to_id.json để các script sau dùng nhất quán
#   ✅ Hỗ trợ --streaming mode (an toàn với RAM, không tải toàn bộ 168GB)
#   ✅ Resume-friendly: skip wav đã tồn tại
# Đầu ra:
#   {output_dir}/wavs/              → wav 24kHz Mono 16-bit PCM
#   {output_dir}/train_list.txt     → wav_path|phoneme|speaker_id
#   {output_dir}/val_list.txt       → wav_path|phoneme|speaker_id
#   {output_dir}/phoneme_vocab.json → {char: id} — đọc n_token từ đây
#   {output_dir}/channel_to_id.json → {"channel_name": int_id, ...}
# Cách chạy:
#   # Bước 0 — Set token (1 lần/session):
#   set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx          # Windows CMD
#   $env:HF_TOKEN="hf_xxxxxxxxxxxxxxxxxxxx"       # PowerShell
#   # Test nhanh 500 mẫu:
#   python data_pipeline/prepare_vivoice.py --max_samples 500
#   # Chạy toàn bộ (streaming, tiết kiệm RAM):
#   python data_pipeline/prepare_vivoice.py --streaming
#   # Chạy toàn bộ (cache về disk, nhanh hơn ở lần 2+):
#   python data_pipeline/prepare_vivoice.py
import os
import sys
import json
import logging
import argparse
import random
import re
from dataclasses import dataclass
from typing import Optional
from pathlib import Path

import yaml
import numpy as np

# CONFIGURATION
@dataclass
class PrepareViVoiceConfig:
    """Cấu hình pipeline chuẩn bị dữ liệu viVoice."""

    # HuggingFace
    hf_dataset_name: str   = "capleaf/viVoice"
    hf_cache_dir:    str   = "D:/HUST_Project/Project_Final/data/hf_cache"
    hf_split:        str   = "train"
    # streaming=True → xử lý shard lần lượt, không cần tải toàn bộ về RAM
    streaming:       bool  = False

    # Đường dẫn output
    output_dir: str = "D:/HUST_Project/Project_Final/data/vivoice_processed"

    # Xử lý Audio
    target_sr:      int   = 24000  # StyleTTS2 chuẩn 24kHz
    min_duration_s: float = 0.5    # Bỏ clip ngắn hơn 0.5 giây
    max_duration_s: float = 20.0   # Bỏ clip dài hơn 20 giây

    # Phoneme filtering
    min_phoneme_length: int = 5
    max_phoneme_length: int = 800

    # Train/Val split
    val_ratio:   float = 0.05
    random_seed: int   = 42

    # Giới hạn mẫu (0 = không giới hạn, dùng để test)
    max_samples: int = 0

    # Tên file output
    train_list_filename:  str = "train_list.txt"
    val_list_filename:    str = "val_list.txt"
    vocab_filename:       str = "phoneme_vocab.json"
    channel_map_filename: str = "channel_to_id.json"

    log_level: str = "INFO"

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "PrepareViVoiceConfig":
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        valid = cls.__dataclass_fields__.keys()
        return cls(**{k: v for k, v in cfg.items() if k in valid})

# LOGGING
def setup_logging(log_level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("PrepareViVoice")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        logger.addHandler(h)
    return logger

# BƯỚC 0: KIỂM TRA HF_TOKEN
def check_hf_token(logger: logging.Logger) -> str:
    """
    Đọc HF_TOKEN từ biến môi trường.
    viVoice là gated dataset → phải có token đã được approve.
    """
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        logger.error("=" * 60)
        logger.error("THIẾU HF_TOKEN — viVoice là gated dataset!")
        logger.error("Chạy lệnh sau rồi chạy lại script:")
        logger.error("  Windows CMD : set HF_TOKEN=hf_xxxxxxxxxxxxxxxxxxxx")
        logger.error("  PowerShell  : $env:HF_TOKEN='hf_xxxxxxxxxxxxxxxxxxxx'")
        logger.error("Lấy token tại: https://huggingface.co/settings/tokens")
        logger.error("Approve tại  : https://huggingface.co/datasets/capleaf/viVoice")
        logger.error("=" * 60)
        sys.exit(1)
    logger.info(f"  ✅ HF_TOKEN OK (****{token[-4:]})")
    return token

# BƯỚC 1: LOAD DATASET
def load_vivoice_dataset(config: PrepareViVoiceConfig,
                         token: str,
                         logger: logging.Logger):
    """
    Tải dataset viVoice.
    - KHÔNG dùng trust_remote_code (deprecated trong phiên bản mới)
    - Token truyền vào qua tham số token= (cách HF khuyến nghị)
    - Hỗ trợ streaming và non-streaming
    """
    logger.info("=" * 60)
    logger.info("BƯỚC 1: LOAD DATASET TỪ HUGGING FACE")
    logger.info("=" * 60)
    logger.info(f"  Dataset  : {config.hf_dataset_name}")
    logger.info(f"  Split    : {config.hf_split}")
    logger.info(f"  Streaming: {config.streaming}")
    logger.info(f"  Cache    : {config.hf_cache_dir}")

    try:
        from datasets import load_dataset
    except ImportError:
        logger.error("Thiếu 'datasets'. Chạy: pip install datasets")
        sys.exit(1)

    os.makedirs(config.hf_cache_dir, exist_ok=True)

    try:
        dataset = load_dataset(
            path=config.hf_dataset_name,
            split=config.hf_split,
            cache_dir=config.hf_cache_dir,
            token=token,
            streaming=config.streaming,
            # trust_remote_code đã bị bỏ — dataset viVoice là parquet chuẩn,
            # không cần loading script tự định nghĩa
        )
    except Exception as e:
        logger.error(f"Lỗi tải dataset: {e}")
        logger.error("Kiểm tra lại:")
        logger.error("  1. HF_TOKEN có đúng không?")
        logger.error("  2. Đã approve điều khoản tại trang dataset chưa?")
        logger.error("  3. Kết nối internet ổn định?")
        sys.exit(1)

    if config.streaming:
        logger.info("  ✅ Streaming mode: dataset sẵn sàng (chưa tải về local)")
    else:
        n = len(dataset)
        logger.info(f"  ✅ Loaded! Tổng mẫu: {n:,}")
        logger.info(f"  Columns: {dataset.column_names}")
        # In thử các channel để xác nhận
        sample_channels = set()
        for i in range(min(200, n)):
            ch = dataset[i].get("channel", None)
            if ch:
                sample_channels.add(str(ch))
        logger.info(f"  Channels (200 mẫu đầu): {sorted(sample_channels)}")

    return dataset

# BƯỚC 2: BUILD CHANNEL → SPEAKER_ID MAP (non-streaming only)
def build_channel_map(dataset,
                      config: PrepareViVoiceConfig,
                      logger: logging.Logger) -> dict:
    """
    Quét toàn bộ dataset, thu thập channel names duy nhất,
    gán integer ID tăng dần.

    Streaming mode: trả về {} — sẽ được build on-the-fly trong vòng lặp chính.
    """
    if config.streaming:
        logger.info("  Streaming mode → channel map build on-the-fly.")
        return {}

    logger.info("=" * 60)
    logger.info("BƯỚC 2: BUILD CHANNEL → SPEAKER_ID MAP")
    logger.info("=" * 60)

    total     = len(dataset)
    log_every = max(1, total // 10)
    channels  = set()

    logger.info(f"  Đang quét {total:,} mẫu...")
    for i in range(total):
        ch = dataset[i].get("channel", None)
        if ch:
            channels.add(str(ch))
        if (i + 1) % log_every == 0:
            logger.info(f"  [{(i+1)/total*100:.0f}%] {i+1:,}/{total:,} | "
                        f"channels: {len(channels)}")

    sorted_channels = sorted(channels)
    channel_to_id   = {ch: i for i, ch in enumerate(sorted_channels)}

    logger.info(f"  ✅ {len(channel_to_id)} speaker/channel:")
    for ch, cid in channel_to_id.items():
        logger.info(f"     [{cid:3d}] {ch}")

    return channel_to_id

# AUDIO UTILITIES
def resample_and_save(audio_array: np.ndarray,
                      src_sr: int,
                      target_sr: int,
                      save_path: str) -> bool:
    """Resample → Mono → Normalize → Lưu 16-bit PCM WAV."""
    try:
        import torchaudio
        import torch
        import soundfile as sf
    except ImportError as e:
        print(f"[ERROR] Thiếu thư viện audio: {e}")
        print("Cài: pip install torchaudio soundfile")
        sys.exit(1)

    wav = torch.from_numpy(audio_array.astype(np.float32))

    # Đảm bảo shape [C, T]
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    elif wav.dim() == 2 and wav.shape[0] > wav.shape[1]:
        wav = wav.T  # Transpose nếu bị [T, C]

    # Chuyển Mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Resample
    if src_sr != target_sr:
        wav = torchaudio.transforms.Resample(src_sr, target_sr)(wav)

    # Peak normalize (an toàn cho 16-bit PCM)
    peak = wav.abs().max()
    if peak > 1e-6:
        wav = wav / peak * 0.95

    sf.write(save_path, wav.squeeze(0).numpy(), target_sr, subtype="PCM_16")
    return True

# TEXT / PHONEME UTILITIES
def clean_text(text: str) -> str:
    """
    Làm sạch text trước khi phonemize:
    - Bỏ ngoặc, dấu nháy, gạch ngang kép
    - Giữ dấu câu cơ bản: , . ? !
    - Giữ Unicode tiếng Việt đầy đủ
    """
    text = re.sub(r'["""\'\'«»\(\)\[\]\{\}]', '', text)
    text = re.sub(r'[-–—]{2,}', ' ', text)
    text = re.sub(r'[-–—]', ' ', text)
    # Giữ: chữ cái (bao gồm Unicode tiếng Việt), khoảng trắng, dấu câu cơ bản
    text = re.sub(
        r'[^\w\s,\.?!'
        r'àáảãạăắặẳẵầấậẩẫâèéẻẽẹêềếệểễìíỉĩịòóỏõọôồốộổỗơờớợởỡùúủũụưừứựửữỳýỷỹỵđ'
        r'ÀÁẢÃẠĂẮẶẲẴẦẤẬẨẪÂÊỀẾỆỂỄÌÍỈĨỊÒÓỎÕỌÔỒỐỘỔỖƠỜỚỢỞỠÙÚỦŨỤƯỪỨỰỬỮỲÝỶỸỴĐ'
        r']', ' ', text
    )
    return re.sub(r'\s+', ' ', text).strip()

def phonemize(text: str, viphoneme_module) -> Optional[str]:
    """clean_text → vi2IPA → trả về chuỗi phoneme hoặc None."""
    try:
        cleaned = clean_text(text)
        if not cleaned:
            return None
        result = viphoneme_module.vi2IPA(cleaned)
        if not result or not result.strip():
            return None
        return result.strip()
    except Exception:
        return None

# BƯỚC 5: BUILD PHONEME VOCABULARY
def build_vocabulary(phoneme_list: list, logger: logging.Logger) -> dict:
    """
    Thu thập ký tự IPA duy nhất → {char: id}.
    Special tokens đặt đầu (PAD=0, UNK=1, BOS=2, EOS=3).
    n_token = len(vocab) → điền vào model_params.n_token.
    """
    logger.info("=" * 60)
    logger.info("BƯỚC 5: BUILD PHONEME VOCABULARY")
    logger.info("=" * 60)

    all_chars = set()
    for ph in phoneme_list:
        if ph:
            all_chars.update(ph)

    logger.info(f"  Ký tự IPA duy nhất: {len(all_chars)}")
    logger.info(f"  Ký tự: {sorted(all_chars)}")

    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]
    vocab = {tok: i for i, tok in enumerate(special_tokens)}
    for i, ch in enumerate(sorted(all_chars)):
        vocab[ch] = len(special_tokens) + i

    n = len(vocab)
    logger.info("")
    logger.info(f"  Special tokens : {len(special_tokens)}")
    logger.info(f"  IPA chars      : {len(all_chars)}")
    logger.info(f"  {'─'*40}")
    logger.info(f"  ✅ n_token = {n}  ← điền vào config_stage*.yaml")
    logger.info(f"  {'─'*40}")
    return vocab

# PIPELINE CHÍNH
def run_pipeline(config: PrepareViVoiceConfig, logger: logging.Logger):
    logger.info("")
    logger.info("   PREPARE VIVOICE — StyleTTS2 Vietnamese Data Pipeline   ")
    logger.info(f"  output_dir  : {config.output_dir}")
    logger.info(f"  target_sr   : {config.target_sr} Hz")
    logger.info(f"  streaming   : {config.streaming}")
    logger.info(f"  max_samples : {config.max_samples or 'Không giới hạn'}")
    logger.info(f"  val_ratio   : {config.val_ratio*100:.0f}%")
    logger.info("")

    random.seed(config.random_seed)
    np.random.seed(config.random_seed)

    wav_dir = os.path.join(config.output_dir, "wavs")
    os.makedirs(wav_dir, exist_ok=True)

    # --- Bước 0: Token ---
    token = check_hf_token(logger)

    # --- Bước 1: Load dataset ---
    logger.info("")
    dataset = load_vivoice_dataset(config, token, logger)

    # --- Bước 2: Channel map (non-streaming) ---
    logger.info("")
    channel_to_id: dict = build_channel_map(dataset, config, logger)

    # --- Bước 3: Load viphoneme ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 3: LOAD VIPHONEME")
    logger.info("=" * 60)
    try:
        import viphoneme
        logger.info("  ✅ viphoneme loaded.")
    except ImportError:
        logger.error("Thiếu 'viphoneme'. Chạy: pip install viphoneme")
        sys.exit(1)

    # --- Bước 4: Xử lý từng mẫu ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 4: RESAMPLE + PHONEMIZE")
    logger.info("=" * 60)

    # Chuẩn bị iterator
    if not config.streaming and config.max_samples > 0:
        dataset = dataset.select(range(min(config.max_samples, len(dataset))))

    total_known = (
        len(dataset) if not config.streaming
        else (config.max_samples if config.max_samples > 0 else None)
    )
    log_every = max(1, (total_known or 10000) // 20)

    logger.info(f"  Mẫu dự kiến: {total_known or 'không rõ (streaming)'}")

    successful_entries  = []   # [(wav_abs_path, phoneme_str, speaker_id)]
    all_phoneme_strings = []
    n_proc = n_ok = n_skip_t = n_skip_a = n_err = 0

    for sample in dataset:
        # Giới hạn max_samples ở streaming mode
        if config.streaming and config.max_samples > 0:
            if n_proc >= config.max_samples:
                break

        idx    = n_proc
        n_proc += 1

        try:
            # --- Text + Phoneme ---
            raw_text = sample.get("text", "")
            if not raw_text or not isinstance(raw_text, str):
                n_skip_t += 1
                continue

            ph_str = phonemize(raw_text, viphoneme)
            if ph_str is None:
                n_skip_t += 1
                continue

            ph_len = len(ph_str)
            if not (config.min_phoneme_length <= ph_len <= config.max_phoneme_length):
                n_skip_t += 1
                continue

            # --- Speaker ID từ cột 'channel' ---
            raw_channel = str(sample.get("channel", "unknown")).strip()
            if raw_channel not in channel_to_id:
                # On-the-fly assignment (streaming mode hoặc channel lạ)
                new_id = len(channel_to_id)
                channel_to_id[raw_channel] = new_id
                logger.info(f"  [Mới] channel='{raw_channel}' → id={new_id}")
            speaker_id = channel_to_id[raw_channel]

            # --- Audio ---
            audio_data = sample.get("audio", None)
            if audio_data is None:
                n_skip_a += 1
                continue

            audio_array = np.array(audio_data["array"], dtype=np.float32)
            src_sr      = int(audio_data["sampling_rate"])
            duration_s  = len(audio_array) / max(src_sr, 1)

            if not (config.min_duration_s <= duration_s <= config.max_duration_s):
                n_skip_a += 1
                continue

            # --- Lưu WAV ---
            wav_filename = f"vivoice_{idx:08d}.wav"
            wav_path     = os.path.join(wav_dir, wav_filename)
            wav_abs_path = str(Path(wav_path).resolve())

            if not os.path.exists(wav_path):  # Resume-friendly
                resample_and_save(audio_array, src_sr, config.target_sr, wav_path)

            # --- Ghi nhận ---
            # Format: wav_path|phoneme|speaker_id (multispeaker meldataset.py)
            successful_entries.append((wav_abs_path, ph_str, speaker_id))
            all_phoneme_strings.append(ph_str)
            n_ok += 1

        except Exception as e:
            n_err += 1
            if n_err <= 5:
                logger.warning(f"  Lỗi idx={idx}: {e}")
            elif n_err == 6:
                logger.warning("  (ẩn lỗi tiếp theo...)")
            continue

        # Tiến độ
        if n_proc % log_every == 0:
            pct = (f"{n_proc/total_known*100:.1f}%"
                   if total_known else f"{n_proc:,} mẫu")
            logger.info(
                f"  [{pct}] ✅ {n_ok:,} | "
                f"⏭ text:{n_skip_t:,} audio:{n_skip_a:,} | ❌ {n_err:,}"
            )

    logger.info("")
    logger.info("  📊 KẾT QUẢ:")
    logger.info(f"     Duyệt     : {n_proc:,}")
    logger.info(f"     Thành công: {n_ok:,}")
    logger.info(f"     Skip text : {n_skip_t:,}")
    logger.info(f"     Skip audio: {n_skip_a:,}")
    logger.info(f"     Lỗi       : {n_err:,}")
    logger.info(f"     Speakers  : {len(channel_to_id)}")

    if n_ok == 0:
        logger.error("Không có mẫu nào thành công!")
        sys.exit(1)

    # --- Bước 5: Vocabulary ---
    logger.info("")
    vocab = build_vocabulary(all_phoneme_strings, logger)

    # --- Bước 6: Train/Val Split ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 6: TRAIN / VAL SPLIT")
    logger.info("=" * 60)

    idxs   = list(range(len(successful_entries)))
    random.shuffle(idxs)
    n_val  = max(1, int(len(successful_entries) * config.val_ratio))

    train_entries = [successful_entries[i] for i in idxs[n_val:]]
    val_entries   = [successful_entries[i] for i in idxs[:n_val]]

    logger.info(f"  Train: {len(train_entries):,} | Val: {len(val_entries):,}")

    # --- Bước 7: Lưu files ---
    logger.info("")
    logger.info("=" * 60)
    logger.info("BƯỚC 7: LƯU FILES")
    logger.info("=" * 60)

    train_path   = os.path.join(config.output_dir, config.train_list_filename)
    val_path     = os.path.join(config.output_dir, config.val_list_filename)
    vocab_path   = os.path.join(config.output_dir, config.vocab_filename)
    channel_path = os.path.join(config.output_dir, config.channel_map_filename)

    # train_list.txt / val_list.txt — format: wav_path|phoneme|speaker_id
    for path, entries in [(train_path, train_entries), (val_path, val_entries)]:
        with open(path, "w", encoding="utf-8") as f:
            for wav_p, ph, spk in entries:
                f.write(f"{wav_p}|{ph}|{spk}\n")
        logger.info(f"  ✅ {path}")

    # phoneme_vocab.json
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_token"       : len(vocab),
            "special_tokens": ["<PAD>", "<UNK>", "<BOS>", "<EOS>"],
            "vocab"         : vocab,
            "note"          : "Điền n_token vào model_params.n_token trong config_stage*.yaml"
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ {vocab_path}")

    # channel_to_id.json
    with open(channel_path, "w", encoding="utf-8") as f:
        json.dump({
            "n_speakers"   : len(channel_to_id),
            "channel_to_id": channel_to_id,
            "note"         : "Dùng trong prepare_ngan_phoneme.py: gán speaker_id=0 cho Bác Ngạn"
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"  ✅ {channel_path}")

    # TỔNG KẾT
    logger.info("")
    logger.info("                         TỔNG KẾT                            ")
    logger.info(f"  wavs/          : {wav_dir}")
    logger.info(f"  train_list.txt : {train_path}")
    logger.info(f"  val_list.txt   : {val_path}")
    logger.info(f"  phoneme_vocab  : {vocab_path}")
    logger.info(f"  channel_map    : {channel_path}")
    logger.info("  ⭐ THÔNG SỐ ĐIỀN VÀO config_stage*.yaml:")
    logger.info(f"     model_params:")
    logger.info(f"       n_token      : {len(vocab)}")
    logger.info(f"       multispeaker : true")
    logger.info("  ⭐ STAGE 3 (fine-tune Bác Ngạn):")
    logger.info("     → speaker_id = 0  (gán cứng trong prepare_ngan_phoneme.py)")

# ENTRY POINT
def main():
    parser = argparse.ArgumentParser(
        description="Chuẩn bị dataset viVoice cho StyleTTS2 tiếng Việt (v2).",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "--config", "-c", type=str, default=None,
        help="Đường dẫn YAML config (tuỳ chọn)"
    )
    parser.add_argument(
        "--max_samples", type=int, default=None,
        help="Giới hạn số mẫu (dùng để test nhanh, ví dụ: --max_samples 500)"
    )
    parser.add_argument(
        "--streaming", action="store_true", default=False,
        help="Dùng streaming mode — không load toàn bộ dataset vào RAM\n"
             "(phù hợp với máy RAM ít; channel map build on-the-fly)"
    )
    args = parser.parse_args()

    if args.config and os.path.exists(args.config):
        config = PrepareViVoiceConfig.from_yaml(args.config)
        print(f"[INFO] Config từ: {args.config}")
    else:
        config = PrepareViVoiceConfig()
        if args.config:
            print(f"[WARN] Không tìm thấy '{args.config}', dùng default.")

    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.streaming:
        config.streaming = True

    logger = setup_logging(config.log_level)
    run_pipeline(config, logger)

if __name__ == "__main__":
    main()