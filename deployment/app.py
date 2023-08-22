import streamlit as st
from streamlit_option_menu import option_menu
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase
from prediction import predict_emotion
import eda
import model_result

st.sidebar.header("Emotion Classification")
st.title("Facial Emotion Classification")

class EmotionDetectionTransformer(VideoTransformerBase):
    def transform(self, frame):
        annotated_frame = predict_emotion(frame)
        return annotated_frame

def main():
    st.title('Emotion Detection App')
    st.write("Press Start")

    webrtc_streamer(key="example", video_transformer_factory=EmotionDetectionTransformer)

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
    main()  # Call the main function for emotion detection
