#!/usr/bin/env python3
# rename_files.py

import re
from pathlib import Path

# map tên tiền tố file (tiếng thường, có gạch dưới) sang dạng CamelCase
PREFIX_MAP = {
    'sau_rieng': 'SauRieng',
    'tao':       'Tao',
    'thanh_long':'ThanhLong'
}

# pattern chung: prefix + "_" + 4 chữ số + .txt
PATTERN = re.compile(r'^(sau_rieng|tao|thanh_long)_(\d{4})\.txt$', re.IGNORECASE)

def rename_files(folder: Path):
    if not folder.is_dir():
        print(f"[ERROR] '{folder}' không phải là thư mục hợp lệ.")
        return

    for file in folder.iterdir():
        if not file.is_file() or file.suffix.lower() != '.txt':
            continue

        m = PATTERN.match(file.name)
        if not m:
            continue

        orig_prefix = m.group(1).lower()
        digits4     = m.group(2)      # 4 chữ số, ví dụ "0123", "1000", "9999"
        # Lấy 3 chữ số cuối để thành số từ 000–999
        new_digits3 = digits4[-3:]    # "0123"→"123", "1000"→"000", "0999"→"999"

        new_prefix = PREFIX_MAP[orig_prefix]
        new_name   = f"{new_prefix}_{new_digits3}.txt"
        new_path   = file.with_name(new_name)

        # nếu file đích đã tồn tại, in cảnh báo và bỏ qua
        if new_path.exists():
            print(f"[WARN] Đã có {new_name}, bỏ qua file {file.name}.")
            continue

        # thực hiện đổi tên
        file.rename(new_path)
        print(f"Đã đổi {file.name} → {new_name}")

if __name__ == "__main__":
    # Nhập đường dẫn thư mục từ người dùng
    folder_input = input("Nhập đường dẫn đến thư mục chứa các file .txt cần đổi tên: ")
    folder_path = Path(folder_input.strip())
    rename_files(folder_path)
