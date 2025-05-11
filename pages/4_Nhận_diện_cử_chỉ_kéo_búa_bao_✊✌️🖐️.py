import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque
from PIL import Image, ImageDraw, ImageFont

import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style
style.set_sidebar_background()


# --- Cấu hình Streamlit ---
st.title("Rock–Paper–Scissors Gesture Recognition")
st.write("Nhấn **Open Camera** để bắt đầu, **Stop Camera** để dừng.")
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

@st.cache_resource
def load_rps_model():
    return load_model('NhanDangCuChi/model.h5')
model = load_rps_model()

CLASS_NAMES = ['Rock', 'Paper', 'Scissors']

if 'pred_queue' not in st.session_state:
    st.session_state.pred_queue = deque(maxlen=5)

if 'run' not in st.session_state:
    st.session_state.run = False

col1, col2 = st.columns(2)
with col1:
    if not st.session_state.run and st.button("Open Camera"):
        st.session_state.run = True
        st.stop()
with col2:
    if st.session_state.run and st.button("Stop Camera"):
        st.session_state.run = False
        st.stop()

placeholder = st.empty()

# Hàm tiền xử lý ROI: resize về 150×150 và chuẩn hóa [0,1]
def preprocess_roi(roi, target_size=(224, 224)):
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = cv2.resize(roi, target_size)
    roi = roi.astype('float32') / 255.0
    return np.expand_dims(roi, axis=0)

def draw_unicode_text(frame, text, pos, font_path='ArialUnicode.ttf', font_size=32, color=(255,255,255)):
    img_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        font = ImageFont.load_default()
    draw.text(pos, text, font=font, fill=color)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

# Cấu hình Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

if st.session_state.run:
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened() and st.session_state.run:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                lm = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    frame, lm, mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(255,255,255), thickness=2, circle_radius=3),
                    mp_drawing.DrawingSpec(color=(0,255,0), thickness=2)
                )

                # Tính bounding box
                h, w, _ = frame.shape
                xs = [p.x * w for p in lm.landmark]
                ys = [p.y * h for p in lm.landmark]
                x1, x2 = int(min(xs)) - 20, int(max(xs)) + 20
                y1, y2 = int(min(ys)) - 20, int(max(ys)) + 20
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                roi = frame[y1:y2, x1:x2]
                if roi.size > 0:
                    x = preprocess_roi(roi)
                    preds = model.predict(x)[0]
                    idx = np.argmax(preds)
                    st.session_state.pred_queue.append(idx)

                    # Lấy nhãn phổ biến nhất trong deque
                    most_common = max(set(st.session_state.pred_queue), key=st.session_state.pred_queue.count)
                    label = CLASS_NAMES[most_common]
                    conf = preds[most_common]

                    # Vẽ khung
                    cv2.rectangle(frame, (x1, y1-35), (x2, y1), (0,0,0), -1)
                    frame = draw_unicode_text(frame,
                                              f"{label} {conf*100:,.1f}%",
                                              (x1+5, y1-30))

            placeholder.image(frame, channels="BGR")

    cap.release()
    placeholder.empty()
