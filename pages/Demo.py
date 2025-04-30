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




if __name__ == '__main__':
    style.set_sidebar_background()
    st.title('Page demo')
    # c1,c2 = st.columns(2)
    # if c1.button('Real Time'):
    #     handlerCamera('')
    # if c2.button('Video'):
    #     handlerVideo()