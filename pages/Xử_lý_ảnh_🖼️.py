import streamlit as st
import time
import numpy as np
import sys
import tkinter
from tkinter import Frame, Tk, BOTH, Text, Menu, END
from tkinter.filedialog import Open, SaveAs
import ThucHanhXuLyAnh.Chapter03 as c3
import ThucHanhXuLyAnh.Chapter04 as c4
import ThucHanhXuLyAnh.Chapter05 as c5
import ThucHanhXuLyAnh.Chapter09 as c9
import cv2
from PIL import Image
import os
import sys
import glob
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import style

st.set_page_config(page_title="Xử lý ảnh số")
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

style.set_sidebar_background()

# Initialize session state for image
if 'imgin' not in st.session_state:
    st.session_state.imgin = None
if 'last_selected_chapter' not in st.session_state:
    st.session_state.last_selected_chapter = None
if 'last_selected_option' not in st.session_state:
    st.session_state.last_selected_option = None

def find_image_file(base_path, option_name):
    """Tìm file ảnh với các định dạng khác nhau"""
    # Danh sách các định dạng ảnh phổ biến
    extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Tìm tất cả các file ảnh có thể
    for ext in extensions:
        image_path = os.path.join(base_path, f"{option_name}{ext}")
        if os.path.exists(image_path):
            return image_path
    
    # Nếu không tìm thấy file với tên chính xác, tìm file có chứa tên option
    for ext in extensions:
        pattern = os.path.join(base_path, f"*{option_name}*{ext}")
        files = glob.glob(pattern)
        if files:
            return files[0]
    
    return None

# Define chapter options and selections in sidebar
chapter_options = ["Chapter 3", "Chapter 4", "Chapter 9"]
selected_chapter = st.sidebar.selectbox("Select Chapter", chapter_options)

# Define chapter-specific options
if selected_chapter == "Chapter 3":
    chapter3_options = ["Negative", "Logarit", "Power", "PiecewiseLinear", "Histogram", "HistEqual",
                        "HistEqualColor", "LocalHist", "HistStat", 
                        "BoxFilter", "LowpassGauss","Threshold", "MedianFilter", "Sharpen", "Gradient"]
    chapter3_selected = st.sidebar.selectbox("Select Operation", chapter3_options)
    
    # Load example image if chapter or option changed
    if (st.session_state.last_selected_chapter != selected_chapter or 
        st.session_state.last_selected_option != chapter3_selected):
        image_path = find_image_file(os.path.join(parent_dir, "ThucHanhXuLyAnh/DIP3E_CH03_Original_Images"), chapter3_selected)
        if image_path:
            st.session_state.imgin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        st.session_state.last_selected_chapter = selected_chapter
        st.session_state.last_selected_option = chapter3_selected

elif selected_chapter == "Chapter 4":
    chapter4_options = ["Spectrum", "RemoveMoire","RemoveInterference","RemoveMoireSimple",
                        "CreateMotion","CreateDemotion","CreateDemotionNoise"]
    chapter4_selected = st.sidebar.selectbox("Select Operation", chapter4_options)
    
    # Load example image if chapter or option changed
    if (st.session_state.last_selected_chapter != selected_chapter or 
        st.session_state.last_selected_option != chapter4_selected):
        image_path = find_image_file(os.path.join(parent_dir, "ThucHanhXuLyAnh/DIP3E_CH04_Original_Images"), chapter4_selected)
        if image_path:
            st.session_state.imgin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        st.session_state.last_selected_chapter = selected_chapter
        st.session_state.last_selected_option = chapter4_selected

elif selected_chapter == "Chapter 9":
    chapter9_options = ["Erosion","Dilation","Boundary","Contour","ConnectedComponent", "CountRice"]
    chapter9_selected = st.sidebar.selectbox("Select Operation", chapter9_options)
    
    # Load example image if chapter or option changed
    if (st.session_state.last_selected_chapter != selected_chapter or 
        st.session_state.last_selected_option != chapter9_selected):
        image_path = find_image_file(os.path.join(parent_dir, "ThucHanhXuLyAnh/DIP3E_CH09_Original_Images"), chapter9_selected)
        if image_path:
            st.session_state.imgin = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        st.session_state.last_selected_chapter = selected_chapter
        st.session_state.last_selected_option = chapter9_selected

# File uploader
image_file = st.file_uploader("Upload Your Image (Optional)", type=['jpg', 'jpeg', 'png', 'bmp', 'tif', 'tiff'])

# Process image if available (either from example or upload)
if image_file is not None:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    st.session_state.imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

if st.session_state.imgin is not None:
    imgin = st.session_state.imgin
    
    # Process image based on selected options
    if selected_chapter == "Chapter 3":
        if chapter3_selected == "Negative":
            processed_image = c3.Negative(imgin)
        elif chapter3_selected == "Logarit":
            processed_image = c3.Logarit(imgin)
        elif chapter3_selected == "Power":
            processed_image = c3.Power(imgin)
        elif chapter3_selected == "PiecewiseLinear":
            processed_image = c3.PiecewiseLinear(imgin)
        elif chapter3_selected == "Histogram":
            processed_image = c3.Histogram(imgin)
        elif chapter3_selected == "HistEqual":
            processed_image = c3.HistEqual(imgin)
        elif chapter3_selected == "HistEqualColor":
            imgin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) 
            processed_image = c3.HistEqualColor(imgin)
        elif chapter3_selected == "LocalHist":
            processed_image = c3.LocalHist(imgin)
        elif chapter3_selected == "HistStat":
            processed_image = c3.HistStat(imgin)
        elif chapter3_selected == "BoxFilter":
            processed_image = c3.MyBoxFilter(imgin)
        elif chapter3_selected == "LowpassGauss":
            processed_image = c3.Image_onLowpassGauss(imgin)
        elif chapter3_selected == "Threshold":
            processed_image = c3.Threshold(imgin)
        elif chapter3_selected == "MedianFilter":
            processed_image = c3.MedianFilter(imgin)
        elif chapter3_selected == "Sharpen":
            processed_image = c3.Sharpen(imgin)
        elif chapter3_selected == "Gradient":
            processed_image = c3.Gradient(imgin)
            
    elif selected_chapter == "Chapter 4":
        if chapter4_selected == "Spectrum":
            processed_image = c4.Spectrum(imgin)
        elif chapter4_selected == "RemoveMoire": 
            processed_image = c4.RemoveMoire(imgin)
        elif chapter4_selected == "RemoveInterference":
            processed_image = c4.RemoveInterference(imgin)
        elif chapter4_selected == "RemoveMoireSimple":
            processed_image = c4.RemoveMoireSimple(imgin)
        elif chapter4_selected == "CreateMotion":
            processed_image = c4.CreateMotion(imgin)
        elif chapter4_selected == "CreateDemotion":
            processed_image = c4.CreateDemotion(imgin)
        elif chapter4_selected == "CreateDemotionNoise":
            processed_image = c4.CreateDemotionNoise(imgin)
            
    elif selected_chapter == "Chapter 9":
        if chapter9_selected == "Erosion":
            processed_image = c9.Erosion(imgin)    
        elif chapter9_selected == "Dilation":
            processed_image = c9.Dilation(imgin)
        elif chapter9_selected == "Boundary":
            processed_image = c9.Boundary(imgin)
        elif chapter9_selected == "Contour":
            processed_image = c9.Contour(imgin)   
        elif chapter9_selected == "ConnectedComponent":
            processed_image = c9.ConnectedComponent(imgin)       
        elif chapter9_selected == "CountRice":
            processed_image = c9.CountRice(imgin)
            
    # Display images
    st.subheader("Original Image and Processed Image")
    if image_file is not None:
        # If image was uploaded, convert to RGB for display
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_UNCHANGED)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        st.image([image_rgb, processed_image], width=350)
    else:
        # If using example image, display directly
        st.image([imgin, processed_image], width=350)

st.button("Re-run")
