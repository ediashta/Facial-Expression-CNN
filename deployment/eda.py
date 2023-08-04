import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px

dist_df = pd.read_csv('./csv/distribution.csv')


st.set_page_config(
    page_title="Facial Emotion Classification",
    layout="wide",
    initial_sidebar_state="expanded",
)

def distribution():
    # distribution plot
    
    st.header("Image Distribution")
    
    fig = px.bar(dist_df, 
                 x="label", y="count", 
                 labels={"label": "Emotions", "count": "Image Count"}, 
                 color="label", template="plotly_white")
    fig.update_layout(height=600)
    
    # Show the plot
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown('''
                
                ''')

def samples():    
    st.header("Image Samples")
    
    sample_img = st.sidebar.selectbox(label="Select Emotion Sample", 
                 options=["Angry", "Disgusted", "Fearful", "Happy", "Neutral", "Sad", "Surprised"])
    
    if sample_img == "Angry":
        st.image('./image/angry.png')
        st.markdown('''
                    This emotion is usually characterized by feelings of anger, frustration, and dissatisfaction towards something perceived as threatening or hurtful.

                    **Physical Signs:**

                    * The outer part of the eyebrows is raised.
                    * The eyes are usually narrowed and sharp.
                    * The muscles around the mouth are tensed.
                    ''')
        
    elif sample_img == "Disgusted":
        st.image('./image/disgusted.png')
        st.markdown('''
                    This emotion is related to feelings of fear or being disturbed by something disliked or considered disgusting.

                    **Physical Signs**:
                    - The eyebrows are raised.
                    - The upper eyelids are raised.
                    - The nose is wrinkled.
                    - Usually, the mouth is slightly tense and slightly raised.

                    ''')
        
    elif sample_img == "Fearful":
        st.image('./image/fearful.png')
        st.markdown('''
                    This emotion involves feelings of fear or anxiety due to the presence of a threat or danger.

                    **Physical Signs**:
                    - Widened eyes
                    - Outer part of the eyebrows slightly lowered
                    - Mouth usually slightly downturned
                    - In this emotion, hands are often used to cover parts of the face.

                    ''')
        
    elif sample_img == "Happy":
        st.image('./image/happy.png')
        st.markdown('''
                    This emotion is usually characterized by feelings of joy, happiness, and contentment. Happy facial expressions include smiling, twinkling eyes, and a tendency to be open and friendly.

                    **Physical Signs**:
                    - Eyes appear sparkling
                    - Mouth is smiling

                    ''')
        
    elif sample_img == "Neutral":
        st.image('./image/neutral.png')
        st.markdown('''
                    Neutral emotion indicates the absence of strong emotions or facial expressions that do not show intense feelings.
                    
                    Physical Signs in this expression are that each **facial feature appears not prominent**.

                    ''')
        
    elif sample_img == "Sad":
        st.image('./image/sad.png')
        st.markdown('''
                    This emotion involves feelings of sadness, disappointment, and loss.

                    Physical Signs:
                    - Eyebrows are usually furrowed.
                    - Eyes become downcast or droopy.
                    - The corners of the mouth slightly downturned.

                    ''')
        
    elif sample_img == "Surprised":
        st.image('./image/surprised.png')
        st.markdown('''
                    This emotion arises when someone experiences an unexpected event or receives surprising information.

                    Physical Signs:
                    - Eyebrows raise entirely.
                    - Eyes widen.
                    - Typically, the mouth opens.

                    ''')

    st.subheader("Mean (Avg.) of Image")
    st.image('./image/mean.png')
    st.markdown('''
                By calculating the mean of image from each class, the following information is obtained:

                * The most prominent expressions are **Happy and Surprised**, easily identifiable from the characteristics of the mouth.
                * **Sad, Fearful, and Neutral** expressions are somewhat challenging to differentiate.
                * **Angry and Disgusted** expressions appear similar due to the characteristics of the eyebrows and mouth.

                ''')

if __name__ == "__main__":
    distribution()
