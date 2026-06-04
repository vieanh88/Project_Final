# Lệnh chạy file này (từ root TTS_StyleTTS2-lite-vi/):
#   python -X utf8 data_pipeline/tools/test_espeak.py
import os
from pathlib import Path

program_files = os.environ.get("PROGRAMFILES", r"C:\Program Files")
os.environ["PHONEMIZER_ESPEAK_LIBRARY"] = str(Path(program_files) / "eSpeak NG" / "libespeak-ng.dll")
os.environ["PHONEMIZER_ESPEAK_PATH"]    = str(Path(program_files) / "eSpeak NG" / "espeak-ng.exe")
os.environ["ESPEAK_DATA_PATH"]          = str(Path(program_files) / "eSpeak NG" / "espeak-ng-data")

from phonemizer.backend import EspeakBackend

backend = EspeakBackend(
    language="vi",
    preserve_punctuation=True,
    with_stress=True,
    language_switch="remove-flags",
)
print(backend.phonemize(["Xin chào, tôi tên là Ngạn."]))