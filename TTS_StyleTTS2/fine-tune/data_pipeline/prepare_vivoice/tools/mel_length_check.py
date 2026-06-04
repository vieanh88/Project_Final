# Kiểm tra độ dài mel spectrogram của các file wav trong train/val list
# Thống kê file có mel length < 80 frames (khoảng 0.8s) để loại bỏ hoặc xử lý riêng
# Script debug lỗi do trong batch có audio/mel quá ngắn
# Khi đưa vào style_encoder hoặc predictor_encoder thì sau nhiều lớp downsample, feature map chỉ còn kích thước khoảng (5 x 4), nhưng layer conv cuối cần kernel (5 x 5)
# Lỗi debug: RuntimeError: Calculated padded input size per channel: (5 x 4). Kernel size: (5 x 5).
#            Kernel size can't be greater than actual input size

import yaml
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
from meldataset import preprocess

config_path = Path("/workspace/Project_Final/TTS_StyleTTS2/fine-tune/config/_processed/config_stage2_processed.yaml")
if not config_path.exists():
    config_path = Path("/workspace/Project_Final/TTS_StyleTTS2/fine-tune/config/config_stage2.yaml")

config = yaml.safe_load(open(config_path, "r", encoding="utf-8"))
data_params = config["data_params"]

root_path = Path(data_params["root_path"])
min_len = int(config.get("min_style_mel_length", 80) or 80)

def get_mel_len(wav_path):
    wave, sr = sf.read(str(wav_path))
    if len(wave.shape) == 2:
        wave = wave[:, 0]
    if sr != 24000:
        wave = librosa.resample(wave, orig_sr=sr, target_sr=24000)
    wave = np.concatenate([np.zeros([5000]), wave, np.zeros([5000])], axis=0)
    mel = preprocess(wave).squeeze()
    return int(mel.size(1))

for key in ["train_data", "val_data"]:
    list_path = Path(data_params[key])
    if not list_path.is_absolute():
        list_path = Path("/workspace/Project_Final/TTS_StyleTTS2/fine-tune") / list_path

    short = []
    total = 0

    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    for idx, line in enumerate(lines):
        total += 1
        rel_wav = line.split("|")[0]
        wav_path = root_path / rel_wav

        try:
            mel_len = get_mel_len(wav_path)
            if mel_len < min_len:
                short.append((mel_len, rel_wav))
        except Exception as e:
            short.append((-1, f"{rel_wav} ERROR: {e}"))

        if (idx + 1) % 1000 == 0:
            print(f"{key}: checked {idx+1}/{len(lines)}")

    print("\n" + "=" * 80)
    print(f"{key}: total={total}, short(<{min_len})={len(short)}")
    print("First 30 short files:")
    for item in short[:30]:
        print(item)