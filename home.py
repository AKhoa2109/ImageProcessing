import streamlit as st
import style
from PIL import Image
import requests
from io import BytesIO

# Thêm CSS để đặt hình nền
st.markdown("""
    <style>
        .stApp {
            background-image: url("https://img.freepik.com/free-vector/futuristic-white-technology-background_23-2148390336.jpg?semt=ais_hybrid&w=740");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }
        .stApp > header {
            background-color: transparent;
        }
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

st.set_page_config(
    page_title="Đồ án cuối kỳ",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# style.py
import streamlit as st

def main():
    style.set_sidebar_background()
    
    # Tạo hai cột cho tiêu đề và logo
    title_col, logo_col = st.columns([3, 1])
    
    with title_col:
        st.write("# Đồ án cuối kỳ")
    
    with logo_col:
        # Tải và hiển thị logo FIT
        response = requests.get("https://fit.hcmute.edu.vn/Resources/Images/SubDomain/fit/logo-cntt2021.png")
        image = Image.open(BytesIO(response.content))
        st.image(image, width=200)
    
    style.set_sidebar_background()
    
    st.markdown("## Sản phẩm")
    st.write("Project cuối kỳ cho môn học xử lý ảnh số DIPR430685.  Thuộc Trường Đại Học Sư Phạm Kỹ Thuật TP.HCM.")

    st.markdown("## Thông tin sinh viên thực hiện")
    # tạo 2 cột
    col1, col2 = st.columns(2)

    with col1:
        st.write("**Họ tên:** Nguyễn Hoàng Anh Khoa")
        st.write("**MSSV:** 22110352")

    with col2:
        st.write("**Họ tên:** Lê Đình Lộc")
        st.write("**MSSV:** 22110369")

    st.markdown("### Chức năng trong bài")
    st.write(
        """
        - 📖 Nhận diện chớp mắt
        - 📖 Nhận diện cảm xúc 
        - 📖 Nhận diện khuôn mặt  
        - 📖 Nhận diện trái cây
        - 📖 Nhận dạng xe
        - 📖 Xử lý ảnh
        """
    )

if __name__ == "__main__":
    main()