import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import time

def report():
    
    df = pd.read_csv('./csv/training_history.csv')
    df.rename(columns={'Unnamed: 0':'epoch'}, inplace=True)
    
    st.header("Model Report")
    
    st.subheader("Performance")
    plot_anim = st.sidebar.selectbox(label='Select Performance Metrics', options=["Accuracy", "Loss"])
       
    
    def performance_plot(data):
        progress_bar = st.sidebar.progress(0)
        status_text = st.sidebar.empty()
        last_rows = [df[data].iloc[0]]
        chart = st.line_chart(last_rows, use_container_width=True, height=400)

        for i in range(1, len(df)):
            new_rows = [df[data].iloc[i]]
            status_text.text(f"{round(i/63 * 100, 2)} % Complete")
            chart.add_rows(new_rows)
            progress_bar.progress(i)
            last_rows = new_rows
            time.sleep(0.05)

        progress_bar.empty()
    if plot_anim == "Accuracy":
        data_plot = ['accuracy', 'val_accuracy']
        performance_plot(data_plot)
    else:
        data_plot = ['loss', 'val_loss']
        performance_plot(data_plot)
        
    st.button("Re-run")
    
    st.markdown('''
                * In this model, it can be observed that the convergence occurs before epoch 20.
                * From epoch 40-50, the model starts to stagnate, prompting a reduction in the learning rate.
                * However, it can be seen that the model is slightly less stable in validation before the learning rate reduction.
                * Based on these observations, we can say that this model is still slightly underfit.
                ''')
        
if __name__ == "__main__":
    report()
