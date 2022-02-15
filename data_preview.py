import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
#from st_aggrid import AgGrid

def Heatmap(data_obj):
    fig = plt.figure(figsize=(16, 6))
    sns.heatmap(data_obj.df.corr(), vmin=-1, vmax=1, annot=True, fmt='.2%').set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12)
    st.pyplot(fig)

def data_preview_run(data_obj):
    st.header("DATA PREVIEW")
    col1, col2 = st.columns(2)
    col3, col4 = st.columns(2)
    
    with col1:
        st.subheader("Original dataframe")
        st.dataframe(data_obj.df)
        st.write(data_obj.df.shape)
        
    with col2:
        st.subheader("Dataframe description")
        st.dataframe(data_obj.df.describe())
    
    with col3:
        st.subheader("Data types")
        st.dataframe(data_obj.df.dtypes)
        
    with col4:
        st.subheader("Correlation")
        st.dataframe(data_obj.df.corr())

    Heatmap(data_obj)
    #AgGrid(data_obj.df)