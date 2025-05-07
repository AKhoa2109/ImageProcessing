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
    # Load the emotion detection model
    model = load_model(MODEL_PATH, compile=False)
    
    # Load the face cascade classifier directly from OpenCV
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
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    # Resize to 48x48
    resized = cv2.resize(gray, (48, 48))
    # Convert to array and normalize
    img_array = img_to_array(resized)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def main():
    st.title("Nhận diện cảm xúc khuôn mặt real-time")
    
    # Add a start/stop button
    start_button = st.button("Bắt đầu/ Dừng")
    
    # Initialize session state for camera
    if 'camera_on' not in st.session_state:
        st.session_state.camera_on = False
    if 'cap' not in st.session_state:
        st.session_state.cap = None
    
    # Toggle camera state when button is clicked
    if start_button:
        if st.session_state.camera_on:
            # Dừng camera
            if st.session_state.cap is not None:
                st.session_state.cap.release()
                st.session_state.cap = None
            st.session_state.camera_on = False
        else:
            # Bắt đầu camera
            try:
                st.session_state.cap = cv2.VideoCapture('NhanDienCamXuc/test.mp4')
                if not st.session_state.cap.isOpened():
                    st.error("Không thể kết nối với camera")
                    st.session_state.camera_on = False
                    return
                st.session_state.camera_on = True
            except Exception as e:
                st.error(f"Lỗi khi khởi tạo camera: {str(e)}")
                st.session_state.camera_on = False
                return
    
    # Create a placeholder for the video feed
    video_placeholder = st.empty()
    
    if st.session_state.camera_on and st.session_state.cap is not None:
        try:
            while st.session_state.camera_on:
                ret, frame = st.session_state.cap.read()
                if not ret:
                    st.error("Không thể đọc frame từ camera")
                    break
                    
                # Convert frame to grayscale for face detection
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Detect faces
                faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # Process each detected face
                for (x, y, w, h) in faces:
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Preprocess face for emotion detection
                    processed_face = preprocess_face(face_roi)
                    
                    # Predict emotion
                    predictions = model.predict(processed_face)
                    emotion_idx = np.argmax(predictions[0])
                    emotion = emotions[emotion_idx]
                    confidence = predictions[0][emotion_idx]
                    
                    # Draw rectangle around face
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Display emotion and confidence
                    text = f"{emotion}: {confidence:.2f}"
                    cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                
                # Convert frame to RGB for Streamlit
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(frame_rgb, channels="RGB")
                
                # Check if stop button is pressed
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