from pathlib import Path
import yaml

cfg_path = Path("D:/Documents/HUST/HUST_Project/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/tools/config.yaml")
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

for key in ["train_data", "val_data"]:
    list_path = cfg.get("data_params", {}).get(key)
    if not list_path:
        print(f"SKIP {key}: empty")
        continue

    p = Path(list_path)
    if not p.exists():
        print(f"SKIP {key}: not found {p}")
        continue

    text = p.read_text(encoding="utf-8", errors="replace")
    new_text = text.replace("\\", "/")

    if new_text == text:
        print(f"OK {key}: no backslash found in {p}")
        continue

    backup = p.with_suffix(p.suffix + ".bak_windows_paths")
    backup.write_text(text, encoding="utf-8")
    p.write_text(new_text, encoding="utf-8")

    print(f"FIXED {key}: {p}")
    print(f"BACKUP     : {backup}")