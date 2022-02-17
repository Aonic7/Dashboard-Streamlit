# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend

# Visualization import section
import seaborn as sns #for plotting
import matplotlib.pyplot as plt #to configure plots
# Importing specific plots
from visualization import Heatmap

def data_preview_run(data_obj):
    """Data Preview main

    Args:
        data_obj (__main__.DataObject): DataObject instance.
    """
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
        st.dataframe(data_obj.df.dtypes.astype(str))
        
    with col4:
        st.subheader("Correlation")
        st.dataframe(data_obj.df.corr())

    # Correlation matrix
    st.subheader("Correlation heatmap")
    Heatmap(data_obj)