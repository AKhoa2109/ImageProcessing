import os
import shutil
import random
import cv2
import matplotlib.pyplot as plt
# Hàm để chia dataset
def split_dataset(data_dir, train_ratio=0.8):
    # Lấy danh sách tất cả các tệp ảnh và nhãn
    images = [f for f in os.listdir(data_dir) if f.endswith('.jpg')]
    
    # Trộn ngẫu nhiên danh sách ảnh
    random.shuffle(images)
    
    # Tính số lượng ảnh cho tập train
    train_size = int(len(images) * train_ratio)
    
    # Chia thành train và val
    train_images = images[:train_size]
    val_images = images[train_size:]
    
    # Sao chép ảnh và nhãn vào các thư mục train và val
    for img_file in train_images:
        shutil.copy(os.path.join(data_dir, img_file), train_dir)
        # Sao chép tệp nhãn nếu tồn tại
        label_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(data_dir, label_file)):
            shutil.copy(os.path.join(data_dir, label_file), train_dir)
    
    for img_file in val_images:
        shutil.copy(os.path.join(data_dir, img_file), val_dir)
        # Sao chép tệp nhãn nếu tồn tại
        label_file = img_file.replace('.jpg', '.txt')
        if os.path.exists(os.path.join(data_dir, label_file)):
            shutil.copy(os.path.join(data_dir, label_file), val_dir)

if __name__ == "__main__":
    # Đường dẫn đến thư mục chứa dữ liệu gốc
    data_dir = 'NhanDangXe/daytime'
    
    # Đường dẫn đến thư mục train và val
    train_dir = 'NhanDangXe/Xe_OK/train/'
    val_dir = 'NhanDangXe/Xe_OK/val/'
    
    # Tạo thư mục train và val nếu chưa tồn tại
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Chia dataset
    split_dataset(data_dir, train_ratio=0.8)