import streamlit as st
import style
from PIL import Image
import requests
from io import BytesIO

st.set_page_config(
    page_title="Đồ án cuối kỳ",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
    <style>
        .stApp {
            background-image: url("https://thetamtru.com.vn/wp-content/uploads/Hinh-anh-background-dep-hoa-tiet-chuyen-nghiep-1536x864.jpg");
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

def show_features():
    st.markdown(
        """
        → 👁️ Nhận diện chớp mắt  
        → 😊 Nhận diện cảm xúc  
        → ✊✌️🖐️ Nhận diện cử chỉ (búa – kéo – bao)  
        → 👤 Nhận diện khuôn mặt  
        → 🍎 Nhận diện trái cây  
        → 🚥 Nhận dạng biển báo giao thông  
        → 🚗 Nhận dạng phương tiện  
        → 🖼️ Xử lý ảnh  
        """,
        unsafe_allow_html=True
    )


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
    show_features()

    k1,k2,k3 = st.columns(3)
    k1.image(Image.open('images/opencv.jpg'))
    k2.image(Image.open('images/streamlit.png'))
    k3.image(Image.open('images/anh1.png'))
    t1,t2,t3 = st.columns(3)
    t1.image(Image.open('images/anh2.jpg'))
    t2.image(Image.open('images/anh3.jpg').resize((256, 256)))
    t3.image(Image.open('images/anh4.png').resize((256, 256)))

if __name__ == "__main__":
    main()