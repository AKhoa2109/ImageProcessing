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
image_file = st._main.file_uploader("Upload Your Image", type=[
                                  'jpg', 'png', 'jpeg', 'tif'])

global imgin
if image_file is not None:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    imgin = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE) 

    chapter_options = ["Chapter 3", "Chapter 4", "Chapter 5", "Chapter 9"]
    selected_chapter = st.sidebar.selectbox("Select an option", chapter_options)


    if selected_chapter == "Chapter 3":
        chapter3_options = ["Negative", "Logarit", "Power", "PiecewiseLinear", "Histogram", "HistEqual",
                                                "HistEqualColor", "LocalHist", "HistStat", 
                                                "BoxFilter", "LowpassGauss","Threshold", "MedianFilter", "Sharpen", "Gradient"]
        
        chapter3_selected = st.sidebar.selectbox("Select an option", chapter3_options)    
        if chapter3_selected  == "Negative":
            processed_image = c3.Negative(imgin)
        elif chapter3_selected  == "Logarit":
            processed_image = c3.Logarit(imgin)
        elif chapter3_selected  == "Power":
            processed_image = c3.Power(imgin)
        elif chapter3_selected  == "PiecewiseLinear":
            processed_image = c3.PiecewiseLinear(imgin)
        elif chapter3_selected  == "Histogram":
            processed_image = c3.Histogram(imgin)
        elif chapter3_selected  == "HistEqual":
            processed_image = c3.HistEqual(imgin)
        elif chapter3_selected  == "HistEqualColor":
            if len(imgin.shape) == 2 or imgin.shape[2] == 1:  # ảnh trắng-đen
                img_bgr = cv2.cvtColor(imgin, cv2.COLOR_GRAY2BGR)
            else:
                img_bgr = imgin
            processed_image = c3.HistEqualColor(img_bgr)
        elif chapter3_selected == "LocalHist":
            # imgin = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            processed_image = c3.LocalHist(imgin)
        elif chapter3_selected  == "HistStat":
            processed_image = c3.HistStat(imgin)
        elif chapter3_selected  == "BoxFilter":
            processed_image = c3.MyBoxFilter(imgin)
        elif chapter3_selected  == "LowpassGauss":
            processed_image = c3.Image_onLowpassGauss(imgin)
        elif chapter3_selected  == "Threshold":
            processed_image = c3.Threshold(imgin)
        elif chapter3_selected  == "MedianFilter":
            processed_image = c3.MedianFilter(imgin)
        elif chapter3_selected  == "Sharpen":
            processed_image = c3.Sharpen(imgin)
        elif chapter3_selected  == "Gradient":
            processed_image = c3.Gradient(imgin)
            
    #Chương 4
    elif selected_chapter == "Chapter 4":
        
        chapter4_options = ["Spectrum", "DrawInferenceFilter", "RemoveMoire","RemoveInterference","RemoveMoireSimple","RemoveInferenceFilter","CreateMotion","CreateDemotion","CreateDemotionNoise"]
        
        chapter4_selected = st.sidebar.selectbox("Select an option", chapter4_options)   
        
        
        if chapter4_selected == "Spectrum":
            processed_image = c4.Spectrum(imgin)
        elif chapter4_selected == "DrawInferenceFilter": # Vẽ bộ lọc giao thoa 
            imgin = Image.new('RGB', (5, 5),  st.get_option("theme.backgroundColor"))
            processed_image = c4.DrawInferenceFilter(imgin)# Xe
        elif chapter4_selected == "RemoveMoire": 
            processed_image = c4.RemoveMoire(imgin) # Xe
        elif chapter4_selected == "RemoveInterference": # Fig0465(a)(cassini).tif
            processed_image = c4.RemoveInterference(imgin)
        elif chapter4_selected == "RemoveMoireSimple":# Quyển sách
            processed_image = c4.RemoveMoireSimple(imgin)
        elif chapter4_selected == "RemoveInferenceFilter": #
            processed_image = c4.RemoveInferenceFilter(imgin)
        elif chapter4_selected == "CreateMotion":
            processed_image = c4.CreateMotion(imgin)
        elif chapter4_selected == "CreateDemotion":
            processed_image = c4.CreateDemotion(imgin)
        elif chapter4_selected == "CreateDemotionNoise":
            processed_image = c4.CreateDemotionNoise(imgin)
                 
    elif selected_chapter == "Chapter 9":
        
        # chapter9_options = ["Erosion", "Dilation","OpeningClosing", "Boundary", "HoleFilling","HoleFillingMouse", "ConnectedComponent", "CountRice"]
        chapter9_options = ["Erosion","Dilation","Boundary","Contour","ConnectedComponent", "CountRice"]
        chapter9_selected = st.sidebar.selectbox("Select an option", chapter9_options)   

        if chapter9_selected  == "Erosion":
            processed_image = c9.Erosion(imgin)    
        elif chapter9_selected  == "Dilation":
            processed_image = c9.Dilation(imgin)
        elif chapter9_selected  == "Boundary":
            processed_image = c9.Boundary(imgin)
        elif chapter9_selected  == "Contour":
            processed_image = c9.Contour(imgin)   
        if chapter9_selected  == "ConnectedComponent":
            processed_image = c9.ConnectedComponent(imgin)       
        elif chapter9_selected  == "CountRice":
            processed_image = c9.CountRice(imgin)          
            
    
    st.subheader("Original Image and Processed Image")
    st.image([imgin, processed_image], width = 350)
st.button("Re-run")
