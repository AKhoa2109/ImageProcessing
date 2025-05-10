import streamlit as st
import argparse
import numpy as np
import cv2 as cv
import joblib
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style

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

def handlerCamera(video):
    FRAME_WINDOW = st.image([])
    cap = None

    if(video==''):
        cap = cv.VideoCapture(0)
    else:
        cap = cv.VideoCapture(video)
    if 'stop' not in st.session_state:
        st.session_state.stop = False
        stop = False

    press = st.button('Stop')
    if press:
        if st.session_state.stop == False:
            st.session_state.stop = True
            cap.release()
    else:
        st.session_state.stop = False

    print('Trang thai nhan Stop', st.session_state.stop)

    if 'frame_stop' not in st.session_state:
        frame_stop = cv.imread('stop.jpg')
        st.session_state.frame_stop = frame_stop
        print('Đã load stop.jpg')

    if st.session_state.stop == True:
        FRAME_WINDOW.image(st.session_state.frame_stop, channels='BGR')

    svc = joblib.load('NhanDangKhuonMat_onnx_Streamlit/model/svc.pkl')
    mydict = ['AnhKhoa', 'DaiThien', 'HuyHung', 'Loc', 'LyHung']
    color = [(0, 255, 0),(255, 0, 0),(0, 0, 255),(255, 255, 0),(0, 255, 255)]

    detector = cv.FaceDetectorYN.create(
        'NhanDangKhuonMat_onnx_Streamlit/model/face_detection_yunet_2023mar.onnx',
        "",
        (320, 320),
        0.9,
        0.3,
        5000)
    
    recognizer = cv.FaceRecognizerSF.create(
    'NhanDangKhuonMat_onnx_Streamlit/model/face_recognition_sface_2021dec.onnx',"")

    tm = cv.TickMeter()

    frameWidth = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    detector.setInputSize([frameWidth, frameHeight])

    dem = 0
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            print('No frames grabbed!')
            break

        # Inference
        tm.start()
        faces = detector.detect(frame) # faces is a tuple
        tm.stop()
        y = 50
        key = cv.waitKey(1)
        if key == 27:
            break
        if faces[1] is not None:
            for face_info in faces[1]:
                face_align = recognizer.alignCrop(frame, face_info)
                face_feature = recognizer.feature(face_align)
                test_predict = svc.predict(face_feature)
                result = mydict[test_predict[0]]

                # Draw results on the input image
                cv.putText(frame,result,(1,y),cv.FONT_HERSHEY_SIMPLEX, 0.5, color[test_predict[0]], 2)
                y = y + 20
                coords = face_info[:-1].astype(np.int32)
                cv.rectangle(frame, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), color[test_predict[0]], 2)

        cv.putText(frame, 'FPS: {:.2f}'.format(tm.getFPS()), (1, 16), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Visualize results
        FRAME_WINDOW.image(frame, channels='BGR')
    cv.destroyAllWindows()
def handlerVideo():
    c1,c2 = st.columns(2)
    video_path = "NhanDangKhuonMat_onnx_Streamlit/trainvideo.mp4"
    with c1:
        video_file = open(video_path, 'rb')
        video_bytes = video_file.read()
        c1.video(video_bytes)
    with c2:
        handlerCamera(video_path)
if __name__ == '__main__':
    style.set_sidebar_background()
    st.title('Nhận dạng khuôn mặt')
    c1,c2 = st.columns(2)
    if c1.button('Real Time'):
        handlerCamera('')
    if c2.button('Video'):
        handlerVideo()