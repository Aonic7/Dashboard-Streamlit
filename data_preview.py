import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def Heatmap(data_obj):
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(data_obj.df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2%').set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig)

def data_preview(data_obj):
    st.header("DATA PREVIEW")
    st.subheader("Original dataframe")
    st.dataframe(data_obj.df)
    st.dataframe(data_obj.df.corr())
    st.dataframe(data_obj.df.describe())
    Heatmap(data_obj)