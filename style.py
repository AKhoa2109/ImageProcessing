import streamlit as st
# Hàm CSS để đặt nền cho sidebar
# Hàm CSS để đặt nền cho sidebar
def set_sidebar_background():
    page_bg_img = """
    <style>
    [data-testid="stSidebar"] > div:first-child {
        background-image: url("https://files.123freevectors.com/wp-content/original/154027-abstract-blue-and-white-background-design.jpg");
        background-position: center;
        background-size: cover;
    }
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)