import streamlit as st
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
from urllib import request
from io import BytesIO


def predict():
    
    emotion_classification_model = load_model('./model/model_fine_tune.h5')
    
    
    url = "https://lh3.googleusercontent.com/0e8O0JPOp_ydq7wqv6kgYz6UpF9w_INfnOLIhnJZBEHFcWIygkuLa3SVghhGYgE0XWzQYBPb6wb1eQFN0pVIAYlzEeNojYuCWg=s0"
    
    def img_url(url):
        res = request.urlopen(url).read()
        img_ori = image.load_img(BytesIO(res))
        img = image.load_img(BytesIO(res), target_size=(48, 48), keep_aspect_ratio=True)
        show_predict(img, img_ori)
        
    def show_predict(img, img_ori):
        col1, col2 = st.columns(2)
        fig = plt.figure()
        plt.imshow(img_ori)
        plt.axis('off')
        col1.pyplot(fig)
        
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0  # Normalize the image

        inf_pred_single = emotion_classification_model.predict(img_array)
        
        data_inf_single = []

        rank = []

        for i in inf_pred_single[0]:
            value = i * 100
            rank.append(value)
            data_inf_single.append(f'{value.round(2)}%')

        rank = (-np.array(rank)).argsort()[:2]
        
        pred_class_single = pd.DataFrame(class_labels).loc[rank][0].tolist()

        prediction_result_single = pd.DataFrame(columns=["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"])
        prediction_result_single.loc[len(prediction_result_single)] = data_inf_single

        prediction_result_single
        
        st.markdown("""
                    <style>
                    .big-font {
                        font-size:30px !important;
                    }
                    </style>
                    """, unsafe_allow_html=True)
        
        col2.write('Prediction Class:')
        col2.markdown(f'<p class="big-font">{pred_class_single[0].capitalize()}</p>', unsafe_allow_html=True)
        
        col2.dataframe(prediction_result_single.set_index(prediction_result_single.columns[0]), use_container_width=True)
    
    class_labels = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]

    st.write('Insert Image URL Below (Make sure face is centered and fitted)')
    
    st.markdown('[Example Image](https://cdn.idntimes.com/content-images/community/2021/12/whatsapp-image-2021-12-02-at-190446-8ecf63e1fa6b5c8c5e9ac43034bc86d3-c563813ea99f16a795ad4c53af10881a_600x400.jpeg)')
    
    col1, col2 = st.columns((9,1))
    url_input = col1.text_input(label="Image Links")
    
    st.markdown(
                """
            <style>
            button {
                height: auto;
                margin-top: 28px !important;
                padding-left: 24px !important;
                padding-right: 24px !important;
            }
            </style>
            """,
                unsafe_allow_html=True,
            )
    pred_button = col2.button(label="Predict")
    
    if pred_button:
        img_url(url_input)
    else:
        img_url(url)
    
    

if __name__ == "__main__":
    predict()
