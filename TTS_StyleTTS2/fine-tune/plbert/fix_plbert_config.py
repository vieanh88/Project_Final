import json
import os

# Đường dẫn tới file config.yml của bạn
config_path = r"D:\HUST_Project\Project_Final\TTS_StyleTTS2\fine-tune\plbert\checkpoints\config.yml"

if not os.path.exists(config_path):
    print(f"Không tìm thấy file: {config_path}")
else:
    with open(config_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Kiểm tra xem đã được bọc chưa, nếu chưa thì bọc lại
    if "model_params" not in data:
        new_data = {"model_params": data}
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(new_data, f, indent=4)
        print("Đã bọc 'model_params' thành công! Cấu trúc file đã chuẩn.")
    else:
        print("File đã có sẵn 'model_params', không cần sửa.")