import streamlit as st
import style


st.set_page_config(
    page_title="Đồ án cuối kỳ",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded",
)

def main():
    style.set_sidebar_background()
    st.write("# Đồ án cuối kỳ")

if __name__ == "__main__":
    main()