import streamlit as st
import base64
from pathlib import Path

# Hàm đọc file ảnh và trả về Base64 string
def img_to_bytes(img_path: str) -> str:
    img_bytes = Path(img_path).read_bytes()
    return base64.b64encode(img_bytes).decode()

# Hàm CSS để đặt nền cho sidebar và chèn logo
def set_sidebar_background():
    # đọc Base64 của logo
    logo_b64 = img_to_bytes("Logo_Trường_Đại_Học_Sư_Phạm_Kỹ_Thuật_TP_Hồ_Chí_Minh.png")

    page_bg_img = f"""
    <style>
    /* background sidebar */
    [data-testid="stSidebar"] > div:first-child {{
        background-image: url("https://slidebazaar.com/wp-content/uploads/2024/08/Free-Professional-Background-PPT-Template.jpg");
        background-position: center;
        background-size: cover;
    }}

    /* content sidebar với padding để chứa logo */
    div[data-testid="stSidebarNav"] {{
      padding-top: 180px;
      position: relative;
    }}

    /* chèn logo vào đầu sidebar */
    div[data-testid="stSidebarNav"]::before {{
      content: "";
      display: block;
      position: absolute;
      top: 16px;
      left: 50%;
      transform: translateX(-50%);
      width: 160px;
      height: 160px;
      background-image: url("https://upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Logo_Tr%C6%B0%E1%BB%9Dng_%C4%90%E1%BA%A1i_H%E1%BB%8Dc_S%C6%B0_Ph%E1%BA%A1m_K%E1%BB%B9_Thu%E1%BA%ADt_TP_H%E1%BB%93_Ch%C3%AD_Minh.png/800px-Logo_Tr%C6%B0%E1%BB%9Dng_%C4%90%E1%BA%A1i_H%E1%BB%8Dc_S%C6%B0_Ph%E1%BA%A1m_K%E1%BB%B9_Thu%E1%BA%ADt_TP_H%E1%BB%93_Ch%C3%AD_Minh.png");
      background-size: contain;
      background-repeat: no-repeat;
      z-index: 1;
    }}


    </style>
    """
    #  background-image: url("data:image/png;base64,{logo_b64}");
    st.markdown(page_bg_img, unsafe_allow_html=True)


def main():
    set_sidebar_background()
    st.write("# Đồ án cuối kỳ")
    # ... phần còn lại của app ...

if __name__ == "__main__":
    main()
