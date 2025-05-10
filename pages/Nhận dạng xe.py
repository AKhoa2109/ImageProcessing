import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import tempfile
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style
style.set_sidebar_background()

st.title('Nhận dạng xe')
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
    if st.session_state["LoadModel"] == True:
        print('Đã load model rồi')
except:
    st.session_state["LoadModel"] = True
    st.session_state["Net"] = cv2.dnn.readNet("NhanDangXe/Xe_OK/train_yolo/yolov8n_xe.onnx")
    # st.session_state["Net"] = cv2.dnn.readNet("NhanDangTraiCay/Yolov8n/yolov8n_traicay.onnx")
    print(st.session_state["LoadModel"])
    print('Load model lần đầu')
     
# filename_classes = 'NhanDangTraiCay/Yolov8n/fruit_detection_classes_yolo.txt'
filename_classes = 'NhanDangXe/Xe_OK/train_yolo/xe_detection_classes_yolo.txt'
mywidth  = 640
myheight = 640
postprocessing = 'yolov8'
background_label_id = -1
backend = 0
target = 0

# Load names of classes
classes = None
if filename_classes:
    with open(filename_classes, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

st.session_state["Net"].setPreferableBackend(0)
st.session_state["Net"].setPreferableTarget(0)
outNames = st.session_state["Net"].getUnconnectedOutLayersNames()

confThreshold = 0.5
nmsThreshold = 0.4
scale = 0.00392
mean = [0, 0, 0]

def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    def drawPred(classId, conf, left, top, right, bottom):
        # Draw a bounding box.
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0))

        label = '%.2f' % conf

        # Print a label of class.
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    layerNames = st.session_state["Net"].getLayerNames()
    lastLayerId = st.session_state["Net"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net"].getLayer(lastLayerId)

    classIds = []
    confidences = []
    boxes = []
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
        # Network produces output blob with a shape NxC where N is a number of
        # detected objects and C is a number of classes + 4 where the first 4
        # numbers are [center_x, center_y, width, height]
        if postprocessing == 'yolov8':
            box_scale_w = frameWidth / mywidth
            box_scale_h = frameHeight / myheight
        else:
            box_scale_w = frameWidth
            box_scale_h = frameHeight

        for out in outs:
            if postprocessing == 'yolov8':
                out = out[0].transpose(1, 0)

            for detection in out:
                scores = detection[4:]
                if background_label_id >= 0:
                    scores = np.delete(scores, background_label_id)
                classId = np.argmax(scores)
                confidence = scores[classId]
                if confidence > confThreshold:
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])
    else:
        print('Unknown output layer type: ' + lastLayer.type)
        exit()

    # NMS is used inside Region layer only on DNN_BACKEND_OPENCV for another backends we need NMS in sample
    # or NMS is required if number of outputs > 1
    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv2.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box  = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)
    return

def process_frame(frame):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Create a 4D blob from a frame
    inpWidth = mywidth if mywidth else frameWidth
    inpHeight = myheight if myheight else frameHeight
    blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)

    # Run a model
    st.session_state["Net"].setInput(blob, scalefactor=scale, mean=mean)
    if st.session_state["Net"].getLayer(0).outputNameToIndex('im_info') != -1:
        frame = cv2.resize(frame, (inpWidth, inpHeight))
        st.session_state["Net"].setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

    outs = st.session_state["Net"].forward(outNames)
    postprocess(frame, outs)
    return frame

# File uploader
file_type = st.radio("Chọn loại file", ["Hình ảnh", "Video"])
uploaded_file = None

if file_type == "Hình ảnh":
    uploaded_file = st.file_uploader("Upload hình ảnh", type=["bmp", "png", "jpg", "jpeg"])
else:
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

col1, col2 = st.columns([1,1])

if uploaded_file is not None:
    if file_type == "Hình ảnh":
        # Xử lý hình ảnh
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        with col1:
            st.image(image, caption="Hình ảnh gốc")
        
        if st.button('Nhận dạng'):
            processed_frame = process_frame(frame.copy())
            color_converted = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(color_converted)
            with col2:
                st.image(pil_image, caption="Kết quả nhận dạng")
    
    else:
        # Xử lý video
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        cap = cv2.VideoCapture(tfile.name)
        
        if st.button('Nhận dạng'):
            stframe1 = col1.empty()
            stframe2 = col2.empty()
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Hiển thị frame gốc
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe1.image(frame_rgb, caption="Video gốc")
                
                # Xử lý và hiển thị frame đã nhận dạng
                processed_frame = process_frame(frame.copy())
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe2.image(processed_rgb, caption="Kết quả nhận dạng")
                
                # Thêm delay nhỏ để video không chạy quá nhanh
                cv2.waitKey(1)
            
            cap.release()
            os.unlink(tfile.name)


