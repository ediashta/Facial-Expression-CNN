import streamlit as st
import eda
import model_result
import prediction
from streamlit_option_menu import option_menu



st.sidebar.header("Emotion Classification")
st.title("Facial Emotion Classification")

with st.sidebar:
    st.write("Ediashta Revindra - FTDS-020")
    selected = option_menu(
        "Menu",
        [
            "Distribution",
            "Image Sample",
            "Model Result",
            "Classification",
        ],
        icons=["bar-chart", "link-45deg", "code-square"],
        menu_icon="cast",
        default_index=0,
    )

if selected == "Distribution":
    eda.distribution()
elif selected == "Image Sample":
    eda.samples()
elif selected == "Model Result":
    model_result.report()   
elif selected == "Classification":
    prediction.predict()
