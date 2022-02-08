import pandas as pd
import streamlit as st

def data_preview():
    st.header("DATA PREVIEW")
    st.subheader("Original dataframe")
    st.dataframe(data_main.df)
