import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style
style.set_sidebar_background()

st.title('Nhận dạng biển báo')
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

try:
    if st.session_state["LoadModelBB"] == True:
        print('Đã load model rồi')
except:
    st.session_state["LoadModelBB"] = True
    st.session_state["NetBB"] = cv2.dnn.readNet("NhanDangTinHieuGiaoThong/train_sign_yolo/best.onnx")
    print(st.session_state["LoadModelBB"])
    print('Load model lần đầu')
     
filename_classes = 'NhanDangTinHieuGiaoThong/train_sign_yolo/detection_classes_yolo.txt'
mywidth  = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

classes = None
if filename_classes:
    with open(filename_classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

st.session_state["NetBB"].setPreferableBackend(0)
st.session_state["NetBB"].setPreferableTarget(0)
outNames = st.session_state["NetBB"].getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

def postprocess(frame, outs):
    """
    Xử lý kết quả từ model và vẽ bounding box
    
    Args:
        frame: Ảnh gốc
        outs: Output từ model
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        """
        Vẽ bounding box và label cho đối tượng được phát hiện
        
        Args:
            classId: ID của lớp
            conf: Độ tin cậy
            left, top, right, bottom: Tọa độ bounding box
        """
        # Vẽ rectangle xanh lá
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        # Tạo label với tên lớp và độ tin cậy
        label = '%.2f' % conf
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        # Vẽ background cho text
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        # Vẽ text
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Lấy thông tin về layer cuối cùng
    layerNames = st.session_state["NetBB"].getLayerNames()
    lastLayerId = st.session_state["NetBB"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["NetBB"].getLayer(lastLayerId)

    # Khởi tạo các list để lưu kết quả
    classIds = []
    confidences = []
    boxes = []

    # Xử lý output từ model
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        # Tính scale cho bounding box
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        # Xử lý từng detection
        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                
                # Lọc theo ngưỡng tin cậy
                if confidence > confThreshold:
                    # Tính toán tọa độ bounding box
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    
                    # Lưu kết quả
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # Áp dụng Non-Maximum Suppression
    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv2.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        
        # Áp dụng NMS cho từng lớp
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    # Vẽ kết quả cuối cùng
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return

img_file_buffer = st.file_uploader("Upload an image", type=["bmp", "png", "jpg", "jpeg"])
col1, col2 = st.columns([1,1])

if img_file_buffer is not None:
    image = Image.open(img_file_buffer)
    # Chuyển sang cv2 để dùng sau này
    frame = np.array(image)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    with col1:  st.image(image, caption="Hình ảnh tải lên")
    if st.button('Predict'):
        if not frame is None:
            frameHeight = frame.shape[0]
            frameWidth = frame.shape[1]

            inpWidth = mywidth if mywidth else frameWidth
            inpHeight = myheight if myheight else frameHeight
            blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)

            st.session_state["NetBB"].setInput(blob, scalefactor=scale, mean=mean)
            if st.session_state["NetBB"].getLayer(0).outputNameToIndex('im_info') != -1:  # Faster-RCNN or R-FCN
                frame = cv2.resize(frame, (inpWidth, inpHeight))
                st.session_state["NetBB"].setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

            outs = st.session_state["NetBB"].forward(outNames)
            postprocess(frame, outs)

            color_coverted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            pil_image = Image.fromarray(color_coverted) 
            with col2:  st.image(pil_image, caption="Kết quả nhận dạng")


