import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from typing import List, Tuple

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
    """Tính toán tỷ lệ khung mắt (Eye Aspect Ratio)"""
    p0, p1, p2, p3, p4, p5 = [np.array(landmarks[i]) for i in eye_idxs]
    vertical1 = np.linalg.norm(p1 - p5)
    vertical2 = np.linalg.norm(p2 - p4)
    horizontal = np.linalg.norm(p0 - p3)
    return (vertical1 + vertical2) / (2.0 * horizontal)

def process_frame(image):
    """Xử lý frame ảnh để phát hiện chớp mắt và vẽ landmarks"""
    h, w = image.shape[:2]
    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:
        rgb_frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0]
            landmarks = [(p.x, p.y) for p in lm.landmark]
            left_ear = calculate_ear(LEFT_EYE_LANDMARKS, landmarks)
            right_ear = calculate_ear(RIGHT_EYE_LANDMARKS, landmarks)
            avg_ear = (left_ear + right_ear) / 2.0

            eye_closed = avg_ear < EAR_THRESHOLD

            # Phát hiện chuyển đổi trạng thái mắt
            if eye_closed and not st.session_state.previous_closed:
                st.session_state.eye_closed = True
            elif not eye_closed and st.session_state.previous_closed:
                st.session_state.blink_counter += 1
                st.session_state.eye_closed = False

            st.session_state.previous_closed = eye_closed

            # Vẽ kết nối mắt
            for connection in [LEFT_EYE_CONNECTIONS, RIGHT_EYE_CONNECTIONS]:
                for start, end in connection:
                    x1 = int(lm.landmark[start].x * w)
                    y1 = int(lm.landmark[start].y * h)
                    x2 = int(lm.landmark[end].x * w)
                    y2 = int(lm.landmark[end].y * h)
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 1)
        return image

def main():
    st.title("Phát hiện chớp mắt thời gian thực 👁️")

    # Khởi tạo session state
    if 'blink_counter' not in st.session_state:
        st.session_state.blink_counter = 0
    if 'previous_closed' not in st.session_state:
        st.session_state.previous_closed = False
    if 'eye_closed' not in st.session_state:
        st.session_state.eye_closed = False

    start = st.button("Bắt đầu phát hiện")
    stop = st.button("Dừng")

    if start:
        cap = cv2.VideoCapture(0)
        img_placeholder = st.empty()
        count_placeholder = st.empty()
        status_placeholder = st.empty()

        while cap.isOpened():
            if stop:
                break
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            processed = process_frame(frame)

            # Hiển thị hình ảnh và thông tin
            img_placeholder.image(processed, channels="BGR", use_container_width=True)
            count_placeholder.markdown(f"**Số lần chớp mắt:** {st.session_state.blink_counter}")
            status = "Đóng mắt" if st.session_state.eye_closed else "Mở mắt"
            status_placeholder.markdown(f"**Trạng thái:** {status}")

        cap.release()

if __name__ == "__main__":
    main()