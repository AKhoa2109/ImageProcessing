import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style
style.set_sidebar_background()
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://slidebazaar.com/wp-content/uploads/2024/08/Free-Professional-Background-PPT-Templates.jpg");
            /* cover → làm đầy, nhưng có thể crop; contain → vừa đủ, giữ nguyên tỉ lệ */
            background-size: contain;     
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;           /* đảm bảo luôn cao tối thiểu 100% chiều cao cửa sổ */
            width: 100%;                 /* đảm bảo luôn rộng 100% */
        }
        .stApp > header,
        .stApp > footer {
            background-color: transparent;
        }
        .stApp > .main > .block-container {
            background-color: rgba(255, 255, 255, 0.9);
            padding: 2rem;
            border-radius: 10px;
            margin: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Cấu hình MediaPipe
mp_face_mesh = mp.solutions.face_mesh

# Định nghĩa kết nối để vẽ
LEFT_EYE_CONNECTIONS = mp_face_mesh.FACEMESH_LEFT_EYE
RIGHT_EYE_CONNECTIONS = mp_face_mesh.FACEMESH_RIGHT_EYE

# Định nghĩa chỉ số landmark để tính EAR
LEFT_EYE_LANDMARKS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_LANDMARKS = [362, 385, 387, 263, 373, 380]

# Hằng số
EAR_THRESHOLD = 0.21  # Ngưỡng tỷ lệ mắt

def calculate_ear(eye_idxs: List[int], landmarks: List[Tuple[float, float]]) -> float:
    """
    Tính toán tỷ lệ khung mắt (Eye Aspect Ratio - EAR)
    
    Args:
        eye_idxs: Danh sách chỉ số của các điểm landmark mắt
        landmarks: Danh sách tọa độ các điểm landmark
        
    Returns:
        float: Giá trị EAR
    """
    # Chuyển đổi các điểm landmark thành mảng numpy
    p0, p1, p2, p3, p4, p5 = [np.array(landmarks[i]) for i in eye_idxs]
    
    # Tính khoảng cách dọc 1 (từ p1 đến p5)
    vertical1 = np.linalg.norm(p1 - p5)
    
    # Tính khoảng cách dọc 2 (từ p2 đến p4)
    vertical2 = np.linalg.norm(p2 - p4)
    
    # Tính khoảng cách ngang (từ p0 đến p3)
    horizontal = np.linalg.norm(p0 - p3)
    
    # Tính EAR: (tổng khoảng cách dọc) / (2 * khoảng cách ngang)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def process_frame(image):
    """
    Xử lý frame ảnh để phát hiện chớp mắt và vẽ landmarks
    
    Args:
        image: Frame ảnh đầu vào
        
    Returns:
        image: Frame ảnh đã được xử lý và vẽ landmarks
    """
    h, w = image.shape[:2]  # Lấy kích thước ảnh
    
    # Khởi tạo FaceMesh với các tham số
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # Chỉ phát hiện 1 khuôn mặt
        refine_landmarks=True,  # Làm mịn landmarks
        min_detection_confidence=0.5,  # Độ tin cậy tối thiểu cho phát hiện
        min_tracking_confidence=0.5  # Độ tin cậy tối thiểu cho tracking
    ) as face_mesh:
        # Chuyển đổi ảnh từ BGR sang RGB
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Xử lý frame với FaceMesh
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            # Lấy landmarks của khuôn mặt đầu tiên
            lm = results.multi_face_landmarks[0]
            
            # Chuyển đổi landmarks thành tọa độ
            landmarks = [(p.x, p.y) for p in lm.landmark]
            
            # Tính EAR cho mắt trái và phải
            left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
            right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
            
            # Tính EAR trung bình
            avg_ear = (left_ear + right_ear) / 2.0

            # Xác định trạng thái mắt
            eye_closed = avg_ear < EAR_THRESHOLD

            # Phát hiện chuyển đổi trạng thái mắt
            if eye_closed and not st.session_state.previous_closed:
                st.session_state.eye_closed = True
            elif not eye_closed and st.session_state.previous_closed:
                st.session_state.blink_counter += 1
                st.session_state.eye_closed = False

            # Cập nhật trạng thái trước đó
            st.session_state.previous_closed = eye_closed

            # Vẽ các kết nối mắt
            for connection in [LEFT_EYE_CONNECTIONS, RIGHT_EYE_CONNECTIONS]:
                for start, end in connection:
                    # Chuyển đổi tọa độ tương đối thành tọa độ pixel
                    x1 = int(lm.landmark[start].x * w)
                    y1 = int(lm.landmark[start].y * h)
                    x2 = int(lm.landmark[end].x * w)
                    y2 = int(lm.landmark[end].y * h)
                    # Vẽ đường kết nối
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
                    
        return image

def main():
    """
    Hàm chính của ứng dụng
    """
    st.title("Phát hiện chớp mắt thời gian thực 👁️")

    # Khởi tạo các biến session state
    if 'blink_counter' not in st.session_state:
        st.session_state.blink_counter = 0  # Đếm số lần chớp mắt
    if 'previous_closed' not in st.session_state:
        st.session_state.previous_closed = False  # Trạng thái mắt trước đó
    if 'eye_closed' not in st.session_state:
        st.session_state.eye_closed = False  # Trạng thái mắt hiện tại

    # Tạo các nút điều khiển
    start = st.button("Bắt đầu phát hiện")
    stop = st.button("Dừng")

    if start:
        # Mở camera
        cap = cv2.VideoCapture(0)
        
        # Tạo các placeholder để hiển thị
        img_placeholder = st.empty()  # Hiển thị ảnh
        count_placeholder = st.empty()  # Hiển thị số lần chớp mắt
        status_placeholder = st.empty()  # Hiển thị trạng thái mắt

        # Xử lý từng frame
        while cap.isOpened():
            if stop:
                break
                
            # Đọc frame từ camera
            ret, frame = cap.read()
            if not ret:
                break
                
            # Lật ảnh để hiển thị như gương
            frame = cv2.flip(frame, 1)
            
            # Xử lý frame
            processed = process_frame(frame)

            # Hiển thị kết quả
            img_placeholder.image(processed, channels="BGR", use_container_width=True)
            count_placeholder.markdown(f"**Số lần chớp mắt:** {st.session_state.blink_counter}")
            status = "Đóng mắt" if st.session_state.eye_closed else "Mở mắt"
            status_placeholder.markdown(f"**Trạng thái:** {status}")

        # Giải phóng camera
        cap.release()

if __name__ == "__main__":
    main()