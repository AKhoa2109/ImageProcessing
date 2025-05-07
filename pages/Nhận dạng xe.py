import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image

st.title('Nhận dạng phương tiện từ video')

# Đường dẫn cố định
model_path = 'NhanDangXe/Xe_OK/train_yolo/yolov8n_xe.onnx'
classes_path = 'NhanDangXe/Xe_OK/train_yolo/xe_detection_classes_yolo.txt'

# Khởi tạo model và classes
if 'net' not in st.session_state:
    # Load classes
    with open(classes_path, 'r') as f:
        classes = f.read().strip().split('\n')
    
    # Load model
    net = cv2.dnn.readNet(model_path)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    
    # Lưu vào session state
    st.session_state.net = net
    st.session_state.classes = classes
    st.session_state.out_names = net.getUnconnectedOutLayersNames()

# Lấy từ session state
net = st.session_state.net
classes = st.session_state.classes
out_names = st.session_state.out_names

# Tham số YOLOv8
inp_width, inp_height = 640, 640
conf_threshold = 0.25
cls_threshold = 0.25
nms_threshold = 0.45

# Hàm vẽ bounding box
def draw_pred(frame, class_id, conf, left, top, right, bottom):
    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
    label = f"{classes[class_id]}: {conf:.2f}"
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (left, top - h - 4), (left + w, top), (0, 255, 0), -1)
    cv2.putText(frame, label, (left, top - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

# Hàm xử lý hậu kỳ
def postprocess(frame, outs):
    frame_h, frame_w = frame.shape[:2]
    detections = []
    for out in outs:
        detections.append(out.reshape(-1, out.shape[-1]))
    detections = np.vstack(detections)

    boxes, confidences, class_ids = [], [], []
    for det in detections:
        obj_conf = det[4]
        if obj_conf < conf_threshold:
            continue
        class_scores = det[5:]
        cls_id = int(np.argmax(class_scores))
        if cls_id >= len(classes):  # Kiểm tra class_id hợp lệ
            continue
        cls_conf = class_scores[cls_id]
        if cls_conf < cls_threshold:
            continue
        conf = float(obj_conf * cls_conf)
        cx, cy, w, h = det[0:4]
        left = int((cx - w/2) * frame_w)
        top = int((cy - h/2) * frame_h)
        width = int(w * frame_w)
        height = int(h * frame_h)
        boxes.append([left, top, width, height])
        confidences.append(conf)
        class_ids.append(cls_id)

    if not boxes:
        return frame

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    if len(indices) > 0:
        for i in indices.flatten():
            l, t, w, h = boxes[i]
            draw_pred(frame, class_ids[i], confidences[i], l, t, l + w, t + h)
    return frame

# Phần xử lý video
video_file = st.file_uploader("Tải lên video", type=['mp4', 'avi', 'mov'])
if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    cap = cv2.VideoCapture(tfile.name)

    col1, col2 = st.columns(2)
    orig_slot = col1.empty()
    proc_slot = col2.empty()

    if st.button('Bắt đầu nhận dạng'):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Xử lý frame
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (inp_width, inp_height), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(out_names)
            processed = postprocess(frame.copy(), outs)
            
            # Hiển thị kết quả
            proc_rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            orig_slot.image(rgb, channels='RGB', use_container_width=True, caption='Video gốc')
            proc_slot.image(proc_rgb, channels='RGB', use_container_width=True, caption='Video đã xử lý')
            
        cap.release()