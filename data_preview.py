import pandas as pd
import streamlit as st

def data_preview(data_obj):
    st.header("DATA PREVIEW")
    st.subheader("Original dataframe")
    st.dataframe(data_obj.df)
