# Chạy script nhỏ này để xem vocab thực tế trong data
from pathlib import Path
from collections import Counter

text = Path("workdir/phoneme_texts.txt").read_text(encoding="utf-8")
chars = Counter(text)

latin = {c: n for c, n in chars.items() if c.isascii() and c.isalpha()}
ipa   = {c: n for c, n in chars.items() if not c.isascii()}

print(f"Latin chars ({len(latin)}): {sorted(latin.keys())}")
print(f"IPA chars   ({len(ipa)}):   {sorted(ipa.keys())[:30]}")