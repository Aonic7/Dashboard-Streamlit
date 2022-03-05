import pandas as pd
import streamlit as st
import numpy as np


def main(data_obj):
    cc1, cc2, cc3 = st.columns(3)
    with cc1:
        st.write(data_obj.df.columns)
    
    with cc2:
        with st.form(key="form"):
            a = []

            

            col_to_change = st.selectbox("Column to change", df.columns)
            new_col_name = st.text_input("New name", value="")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button:
            df = df.rename(columns={col_to_change: new_col_name})

    


if __name__ == "__main__":
    main()