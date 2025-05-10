import streamlit as st
import numpy as np
import cv2
import tempfile
import threading
import queue
import time
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
            background-size: contain;     
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
            min-height: 100vh;
            width: 100%;
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

st.title('Nhận dạng phương tiện từ video (Tối ưu cho video nhiều xe)')
# Khởi tạo các biến session_state quan trọng trước
if 'stop_event' not in st.session_state:
    st.session_state.stop_event = threading.Event()
if 'frame_queue' not in st.session_state:
    st.session_state.frame_queue = queue.Queue(maxsize=30)
if 'running' not in st.session_state:
    st.session_state.running = False
# Đường dẫn cố định
model_path = 'NhanDangXe/Xe_OK/train_yolo/yolov8n_xe.onnx'
classes_path = 'NhanDangXe/Xe_OK/train_yolo/xe_detection_classes_yolo.txt'

# Cấu hình hiệu suất
FRAME_SKIP = 2                 # Xử lý 1 trong 2 khung hình
DISPLAY_SIZE = (640, 360)      # Kích thước hiển thị
CONF_THRESHOLD = 0.5           # Ngưỡng tin cậy
CLS_THRESHOLD = 0.5            # Ngưỡng phân loại
NMS_THRESHOLD = 0.4            # Ngưỡng NMS

# Khởi tạo model và classes
if 'net' not in st.session_state:
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    net = cv2.dnn.readNet(model_path)
    
    # Tự động phát hiện GPU
    if cv2.cuda.getCudaEnabledDeviceCount() > 0:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        st.success("Đang sử dụng tăng tốc GPU")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        st.warning("Đang chạy trên CPU - Hiệu năng có thể bị ảnh hưởng")

    st.session_state.net = net
    st.session_state.classes = classes
    st.session_state.out_names = net.getUnconnectedOutLayersNames()
    st.session_state.running = False
    st.session_state.frame_queue = queue.Queue(maxsize=30)
    st.session_state.stop_event = threading.Event()

net = st.session_state.net
classes = st.session_state.classes
out_names = st.session_state.out_names

def draw_pred(frame, class_id, conf, left, top, right, bottom):
    color = (0, 255, 0)
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    label = f"{classes[class_id]}: {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (left, top - h - 4), (left + w, top), color, -1)
    cv2.putText(frame, label, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

def postprocess(frame, outs):
    frame_h, frame_w = frame.shape[:2]
    detections = []
    for out in outs:
        detections.append(out.reshape(-1, out.shape[-1]))
    detections = np.vstack(detections)

    boxes, confidences, class_ids = [], [], []
    for det in detections:
        obj_conf = det[4]
        if obj_conf < CONF_THRESHOLD:
            continue
        class_scores = det[5:]
        cls_id = np.argmax(class_scores)
        if cls_id >= len(classes):
            continue
        cls_conf = class_scores[cls_id]
        if cls_conf < CLS_THRESHOLD:
            continue
        conf = obj_conf * cls_conf
        cx, cy, w, h = det[0:4]
        left = int((cx - w/2) * frame_w)
        top = int((cy - h/2) * frame_h)
        width = int(w * frame_w)
        height = int(h * frame_h)
        boxes.append([left, top, width, height])
        confidences.append(float(conf))
        class_ids.append(int(cls_id))

    counts = {i:0 for i in range(len(classes))}
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
    
    if len(indices) > 0:
        for i in indices.flatten():
            l, t, w, h = boxes[i]
            cls_id = class_ids[i]
            counts[cls_id] += 1
            draw_pred(frame, cls_id, confidences[i], l, t, l + w, t + h)

    # Vẽ thống kê
    y_offset = 30
    for idx, count in counts.items():
        if count > 0:
            text = f"{classes[idx]}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            y_offset += 30
    
    return frame

def process_video(temp_filename):
    cap = cv2.VideoCapture(temp_filename)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
    
    frame_count = 0
    while cap.isOpened() and not st.session_state.stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % FRAME_SKIP != 0:
            continue
        
        # Xử lý frame
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(out_names)
        processed = postprocess(frame.copy(), outs)
        
        # Giảm kích thước để hiển thị
        processed = cv2.resize(processed, DISPLAY_SIZE)
        processed_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        try:
            st.session_state.frame_queue.put_nowait((frame, processed_rgb))
        except queue.Full:
            pass
    
    cap.release()
    st.session_state.stop_event.set()

# Phần giao diện
video_file = st.file_uploader("Tải lên video", type=['mp4', 'avi', 'mov'])
col1, col2 = st.columns(2)
orig_slot = col1.empty()
proc_slot = col2.empty()

if video_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(video_file.read())
        temp_filename = tfile.name

    control_col = st.columns(3)
    with control_col[0]:
        if st.button('Bắt đầu nhận dạng'):
            st.session_state.running = True
            st.session_state.stop_event.clear()
            threading.Thread(target=process_video, args=(temp_filename,)).start()
    
    with control_col[1]:
        if st.button('Dừng nhận dạng'):
            st.session_state.running = False
            st.session_state.stop_event.set()

    while not st.session_state.stop_event.is_set():
        try:
            orig_frame, processed_frame = st.session_state.frame_queue.get_nowait()
            
            # Hiển thị frame gốc
            orig_frame = cv2.resize(orig_frame, DISPLAY_SIZE)
            orig_rgb = cv2.cvtColor(orig_frame, cv2.COLOR_BGR2RGB)
            orig_slot.image(orig_rgb, channels='RGB', use_container_width=True, caption='Video gốc')
            
            # Hiển thị frame đã xử lý
            proc_slot.image(processed_frame, channels='RGB', use_container_width=True, caption='Video đã xử lý')
            
        except queue.Empty:
            time.sleep(0.01)
    
    try:
        os.unlink(temp_filename)
    except PermissionError:
        st.warning("Hệ thống sẽ tự dọn dẹp file tạm sau")

if not st.session_state.running:
    orig_slot.empty()
    proc_slot.empty()