import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import sys
import tempfile
from collections import defaultdict
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

class VehicleTracker:
    def __init__(self):
        self.tracked_vehicles = {}  # Dictionary lưu thông tin xe đang theo dõi
        self.next_id = 0  # ID cho xe mới
        self.counted_vehicles = set()  # Set lưu ID các xe đã đếm
        self.class_counts = defaultdict(int)  # Đếm số lượng từng loại xe
        self.max_age = 30  # Số frame tối đa một xe có thể không xuất hiện
        self.min_hits = 3  # Số lần xuất hiện tối thiểu để xác nhận là xe mới
        self.iou_threshold = 0.3  # Ngưỡng IoU để xác định là cùng một xe
        self.newly_counted = set()  # Set lưu ID các xe vừa được đếm

    def calculate_iou(self, box1, box2):
        """Tính toán IoU giữa hai bounding box"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[0] + box1[2], box2[0] + box2[2])
        y2 = min(box1[1] + box1[3], box2[1] + box2[3])

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        box1_area = box1[2] * box1[3]
        box2_area = box2[2] * box2[3]
        union = box1_area + box2_area - intersection

        return intersection / union if union > 0 else 0

    def update(self, detections):
        """
        Cập nhật trạng thái tracking và đếm xe mới
        Args:
            detections: List các tuple (class_id, confidence, box)
        Returns:
            class_counts: Dictionary chứa số lượng từng loại xe
            newly_counted: Set chứa ID các xe vừa được đếm
        """
        # Reset newly_counted mỗi frame
        self.newly_counted.clear()
        
        # Cập nhật tuổi của các xe đang theo dõi
        for vehicle_id in list(self.tracked_vehicles.keys()):
            self.tracked_vehicles[vehicle_id]['age'] += 1
            if self.tracked_vehicles[vehicle_id]['age'] > self.max_age:
                del self.tracked_vehicles[vehicle_id]

        # Xử lý các detection mới
        for class_id, conf, box in detections:
            matched = False
            for vehicle_id, vehicle in self.tracked_vehicles.items():
                if self.calculate_iou(box, vehicle['box']) > self.iou_threshold:
                    # Cập nhật thông tin xe đã theo dõi
                    self.tracked_vehicles[vehicle_id]['box'] = box
                    self.tracked_vehicles[vehicle_id]['age'] = 0
                    self.tracked_vehicles[vehicle_id]['hits'] += 1
                    
                    # Nếu xe đã xuất hiện đủ số lần và chưa được đếm
                    if (vehicle_id not in self.counted_vehicles and 
                        self.tracked_vehicles[vehicle_id]['hits'] >= self.min_hits):
                        self.counted_vehicles.add(vehicle_id)
                        self.class_counts[classes[class_id]] += 1
                        self.newly_counted.add(vehicle_id)  # Thêm vào danh sách xe mới đếm
                    
                    matched = True
                    break

            if not matched:
                # Thêm xe mới vào tracking
                self.tracked_vehicles[self.next_id] = {
                    'box': box,
                    'age': 0,
                    'hits': 1
                }
                self.next_id += 1

        return self.class_counts, self.newly_counted

def drawPred(frame, classId, conf, left, top, right, bottom, is_newly_counted=False):
    """
    Hàm vẽ bounding box và nhãn cho đối tượng được nhận dạng
    Args:
        frame: Frame cần vẽ
        classId: ID của lớp
        conf: Độ tin cậy
        left, top, right, bottom: Tọa độ bounding box
        is_newly_counted: True nếu là xe vừa được đếm
    """
    # Vẽ hình chữ nhật bao quanh đối tượng với màu khác nhau
    color = (0, 0, 255) if is_newly_counted else (0, 255, 0)  # Đỏ cho xe mới, xanh cho xe cũ
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

    # Tạo nhãn với độ tin cậy
    label = '%.2f' % conf

    # Thêm tên lớp vào nhãn
    if classes:
        assert(classId < len(classes))
        label = '%s: %s' % (classes[classId], label)

    # Vẽ nền cho text
    labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
    cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

def postprocess(frame, outs, tracker):
    """
    Hàm xử lý kết quả từ model và vẽ bounding box
    Args:
        frame: Ảnh đầu vào
        outs: Kết quả từ model
        tracker: VehicleTracker object
    Returns:
        frame: Frame đã vẽ bounding box
        detections: List các detection (class_id, confidence, box)
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]
    detections = []

    def drawPred(classId, conf, left, top, right, bottom, is_newly_counted=False):
        """
        Hàm vẽ bounding box và nhãn cho đối tượng được nhận dạng
        Args:
            is_newly_counted: True nếu là xe vừa được đếm
        """
        # Vẽ hình chữ nhật bao quanh đối tượng với màu khác nhau
        color = (0, 0, 255) if is_newly_counted else (0, 255, 0)  # Đỏ cho xe mới, xanh cho xe cũ
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)

        # Tạo nhãn với độ tin cậy
        label = '%.2f' % conf

        # Thêm tên lớp vào nhãn
        if classes:
            assert(classId < len(classes))
            label = '%s: %s' % (classes[classId], label)

        # Vẽ nền cho text
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.rectangle(frame, (left, top - labelSize[1]), (left + labelSize[0], top + baseLine), (255, 255, 255), cv2.FILLED)
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    # Lấy thông tin về layer cuối cùng
    layerNames = st.session_state["Net"].getLayerNames()
    lastLayerId = st.session_state["Net"].getLayerId(layerNames[-1])
    lastLayer = st.session_state["Net"].getLayer(lastLayerId)

    # Khởi tạo các list để lưu kết quả
    classIds = []
    confidences = []
    boxes = []

    # Xử lý kết quả từ model
    if lastLayer.type == 'Region' or postprocessing == 'yolov8':
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
                    # Tính toán tọa độ bounding box
                    center_x = int(detection[0] * box_scale_w)
                    center_y = int(detection[1] * box_scale_h)
                    width = int(detection[2] * box_scale_w)
                    height = int(detection[3] * box_scale_h)
                    left = int(center_x - width / 2)
                    top = int(center_y - height / 2)
                    classIds.append(classId)
                    confidences.append(float(confidence))
                    boxes.append([left, top, width, height])

    # Áp dụng Non-Maximum Suppression để loại bỏ các box trùng lặp
    if len(outNames) > 1 or (lastLayer.type == 'Region' or postprocessing == 'yolov8') and 0 != cv2.dnn.DNN_BACKEND_OPENCV:
        indices = []
        classIds = np.array(classIds)
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        unique_classes = set(classIds)
        for cl in unique_classes:
            class_indices = np.where(classIds == cl)[0]
            conf = confidences[class_indices]
            box = boxes[class_indices].tolist()
            nms_indices = cv2.dnn.NMSBoxes(box, conf, confThreshold, nmsThreshold)
            indices.extend(class_indices[nms_indices])
    else:
        indices = np.arange(0, len(classIds))

    # Vẽ các bounding box cuối cùng
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        detections.append((classIds[i], confidences[i], box))
    
    return frame, detections

def process_frame(frame, tracker):
    """
    Hàm xử lý một frame (ảnh hoặc frame video)
    Args:
        frame: Frame cần xử lý
        tracker: VehicleTracker object
    Returns:
        frame: Frame đã được xử lý với các bounding box
        class_counts: Dictionary chứa số lượng từng loại xe
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Tạo blob từ frame
    inpWidth = mywidth if mywidth else frameWidth
    inpHeight = myheight if myheight else frameHeight
    blob = cv2.dnn.blobFromImage(frame, size=(inpWidth, inpHeight), swapRB=True, ddepth=cv2.CV_8U)

    # Chạy model
    st.session_state["Net"].setInput(blob, scalefactor=scale, mean=mean)
    if st.session_state["Net"].getLayer(0).outputNameToIndex('im_info') != -1:
        frame = cv2.resize(frame, (inpWidth, inpHeight))
        st.session_state["Net"].setInput(np.array([[inpHeight, inpWidth, 1.6]], dtype=np.float32), 'im_info')

    outs = st.session_state["Net"].forward(outNames)
    frame, detections = postprocess(frame, outs, tracker)
    class_counts, newly_counted = tracker.update(detections)

    # Vẽ lại các bounding box với màu tương ứng
    for class_id, conf, box in detections:
        left, top, width, height = box
        # Tìm vehicle_id tương ứng với box này
        for vehicle_id, vehicle in tracker.tracked_vehicles.items():
            if tracker.calculate_iou(box, vehicle['box']) > tracker.iou_threshold:
                is_newly_counted = vehicle_id in newly_counted
                drawPred(frame, class_id, conf, left, top, left + width, top + height, is_newly_counted)
                break

    return frame, class_counts

# Tạo giao diện upload file
file_type = st.radio("Chọn loại file", ["Hình ảnh", "Video"])
uploaded_file = None

# Tạo uploader tương ứng với loại file được chọn
if file_type == "Hình ảnh":
    uploaded_file = st.file_uploader("Upload hình ảnh", type=["bmp", "png", "jpg", "jpeg"])
else:
    uploaded_file = st.file_uploader("Upload video", type=["mp4", "avi", "mov"])

# Tạo 2 cột để hiển thị ảnh/video gốc và kết quả
col1, col2 = st.columns([1,1])

if uploaded_file is not None:
    # Khởi tạo tracker
    tracker = VehicleTracker()
    
    if file_type == "Hình ảnh":
        # Xử lý hình ảnh
        image = Image.open(uploaded_file)
        frame = np.array(image)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        
        # Hiển thị ảnh gốc
        with col1:
            st.image(image, caption="Hình ảnh gốc")
        
        # Xử lý và hiển thị kết quả khi nhấn nút
        if st.button('Nhận dạng'):
            processed_frame, class_counts = process_frame(frame.copy(), tracker)
            
            # Hiển thị số lượng từng loại xe
            st.markdown("### Kết quả nhận dạng:")
            for vehicle_type, count in class_counts.items():
                st.markdown(f"- **{vehicle_type}**: {count} xe")
            
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
            # Tạo placeholder cho video gốc và kết quả
            stframe1 = col1.empty()
            stframe2 = col2.empty()
            
            # Tạo placeholder cho hiển thị số lượng
            count_placeholder = st.empty()
            
            # Xử lý từng frame của video
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Hiển thị frame gốc
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                stframe1.image(frame_rgb, caption="Video gốc")
                
                # Xử lý và hiển thị frame đã nhận dạng
                processed_frame, class_counts = process_frame(frame.copy(), tracker)
                processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                stframe2.image(processed_rgb, caption="Kết quả nhận dạng")
                
                # Hiển thị số lượng
                count_text = "### Kết quả nhận dạng:\n"
                for vehicle_type, count in class_counts.items():
                    count_text += f"- **{vehicle_type}**: {count} xe\n"
                count_placeholder.markdown(count_text)
                
                # Thêm delay nhỏ để video không chạy quá nhanh
                cv2.waitKey(1)
            
            # Giải phóng tài nguyên
            cap.release()
            os.unlink(tfile.name)