import os
import re

# Các phần mở rộng cho file txt và ảnh
ALLOWED_EXTENSIONS = {'.txt', '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff'}

def rename_files_in_directory(directory_path):
    """
    Đổi tên các file có tên dạng Buoixxx hoặc Manxxx kèm phần mở rộng txt hoặc ảnh
    Buoixxx -> Buoi_xxx.ext
    Manxxx  -> Man_xxx.ext
    """
    # Kiểm tra đường dẫn
    if not os.path.isdir(directory_path):
        print(f"Đường dẫn không hợp lệ: {directory_path}")
        return

    # Regex chỉ cho phần tên chính (không bao gồm extension)
    name_pattern = re.compile(r"^(ThanhLong)(\d{3})$")

    for filename in os.listdir(directory_path):
        base, ext = os.path.splitext(filename)
        ext = ext.lower()

        # Bỏ qua nếu không phải txt hoặc ảnh
        if ext not in ALLOWED_EXTENSIONS:
            continue

        match = name_pattern.match(base)
        if match:
            prefix, number = match.group(1), match.group(2)
            new_name = f"{prefix}_{number}{ext}"
            old_path = os.path.join(directory_path, filename)
            new_path = os.path.join(directory_path, new_name)

            # Tránh ghi đè nếu tên mới đã tồn tại
            if os.path.exists(new_path):
                print(f"Bỏ qua: '{new_name}' đã tồn tại.")
                continue

            # Đổi tên file
            os.rename(old_path, new_path)
            print(f"Đã đổi '{filename}' thành '{new_name}'")

if __name__ == "__main__":
    dir_path = input("Nhập đường dẫn thư mục chứa file cần chuyển (txt và ảnh): ")
    rename_files_in_directory(dir_path)
