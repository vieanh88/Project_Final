import re

input_file = "workdir/phoneme_texts.txt"
output_file = "workdir/phoneme_texts_clean.txt"

# Lọc các dòng chứa tiếng Hàn, Trung, và các chữ cái tiếng Việt nguyên gốc bị lọt
noise_pattern = re.compile(r'[版輪망실와ịớờủ⁶]')
dropped_count = 0

with open(input_file, 'r', encoding='utf-8') as fin, \
     open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        if noise_pattern.search(line):
            dropped_count += 1
            print(f"Đã xóa dòng: {line.strip()}")
        else:
            fout.write(line)

print(f"\nĐã xóa tổng cộng {dropped_count} dòng chứa noise.")