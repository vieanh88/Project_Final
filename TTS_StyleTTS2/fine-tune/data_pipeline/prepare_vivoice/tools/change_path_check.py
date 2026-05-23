from pathlib import Path
import yaml

cfg = yaml.safe_load(Path("D:/Documents/HUST/HUST_Project/Project_Final/TTS_StyleTTS2/fine-tune/data_pipeline/prepare_vivoice/tools/config.yaml").read_text(encoding="utf-8"))

for key in ["train_data", "val_data"]:
    p = Path(cfg["data_params"][key])
    lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
    bad = [x for x in lines if "\\" in x]
    print(key, "backslash lines:", len(bad))
    if bad[:3]:
        print("\n".join(bad[:3]))