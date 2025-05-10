import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Xác định đường dẫn tuyệt đối tới thư mục hiện tại (nơi chứa file script này)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn tuyệt đối tới model
MODEL_PATH = os.path.join(BASE_DIR, '..', 'NhanDienCamXuc', 'emoji_detection.h5')

# Kiểm tra sự tồn tại của file model
if not os.path.exists(MODEL_PATH):
    st.error(f"Không tìm thấy file model: {MODEL_PATH}")
    st.stop()

try:
   
   #load_model: tải mô hình Keras đã huấn luyện, dùng để dự đoán cảm xúc.
    model = load_model(MODEL_PATH, compile=False)
    
    # CascadeClassifier: thuật toán Haar Cascade có sẵn trong OpenCV để phát hiện mặt.
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        st.error("Không thể load cascade classifier từ OpenCV")
        st.stop()
except Exception as e:
    st.error(f"Lỗi khi load model hoặc cascade: {str(e)}")
    st.stop()

# Define emotion labels
emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face_img):
    # Chuyển sang ảnh xám
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Reszie kích thước 48x48
    resized = cv2.resize(gray, (48, 48))
    # Chuyển thành mảng NumPy 
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def main():
    st.title("Nhận diện cảm xúc khuôn mặt real-time")
    
    start_button = st.button("Bắt đầu/ Dừng")
    
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    
    if start_button:
        if st.session_state.camera_on:
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_on = False
        else:
            try:
                st.session_state.cap = cv2.VideoCapture('NhanDienCamXuc/video.mp4')
                if not st.session_state.cap.isOpened():
                    st.error("Không thể kết nối với camera")
                    st.session_state.camera_on = False
                    return
                st.session_state.camera_on = True
            except Exception as e:
                st.error(f"Lỗi khi khởi tạo camera: {str(e)}")
                st.session_state.camera_on = False
                return
    
    video_placeholder = st.empty()
    
    if st.session_state.camera_on and st.session_state.cap is not None:
        try:
            # Đọc tuần tự từng frame từ video.
            while st.session_state.camera_on:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Không thể đọc frame từ camera")
                    break
                    
                # Chuyển sang xám và phát hiện mặt.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
        
                for (x, y, w, h) in faces:
                   
                    face_roi = frame[y:y+h, x:x+w]
                    
                   
                    processed_face = preprocess_face(face_roi)
                    
              
                    predictions = model.predict(processed_face)
                    emotion_idx = np.argmax(predictions[0])
                    emotion = emotions[emotion_idx]
                    confidence = predictions[0][emotion_idx]
                    
                    # Vẽ hình chữ nhật quanh mặt và ghi nhãn cảm xúc.
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    text = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
           
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(frame_rgb, channels="RGB")
                
                if not st.session_state.camera_on:
                    break
        except Exception as e:
            st.error(f"Lỗi khi xử lý camera: {str(e)}")
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_on = False
    else:
        video_placeholder.info("Nhấn 'Bắt đầu/ Dừng' để bắt đầu nhận diện cảm xúc")

if __name__ == "__main__":
    main() 