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
min_len = int(config.get("min_style_mel_length", 100) or 100)

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

    out_path = list_path.with_name(list_path.stem + f"_minmel{min_len}" + list_path.suffix)
    bad_path = list_path.with_name(list_path.stem + f"_short_minmel{min_len}" + list_path.suffix)

    kept = []
    removed = []

    with open(list_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = [l.strip() for l in f if l.strip()]

    for idx, line in enumerate(lines):
        rel_wav = line.split("|")[0]
        wav_path = root_path / rel_wav

        try:
            mel_len = get_mel_len(wav_path)
            if mel_len >= min_len:
                kept.append(line)
            else:
                removed.append(f"{line}|mel_len={mel_len}")
        except Exception as e:
            removed.append(f"{line}|ERROR={e}")

        if (idx + 1) % 1000 == 0:
            print(f"{key}: checked {idx+1}/{len(lines)}")

    out_path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    bad_path.write_text("\n".join(removed) + "\n", encoding="utf-8")

    print("\n" + "=" * 80)
    print(f"{key}")
    print(f"input   : {list_path}")
    print(f"kept    : {len(kept)} -> {out_path}")
    print(f"removed : {len(removed)} -> {bad_path}")