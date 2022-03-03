from ast import Not
from code import interact
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype

def import_dset(data_obj):
    try:
        rg_df = pd.read_csv(
            'Smoothing_and_Filtering//Preprocessing Dataset.csv', index_col=None)

    except:
        st.error("""You did not smooth of filter the data.
                    Please go to 'Smoothing and filtering' and finalize your results.
                    Otherwise, the default dataset would be used!
                    """)
        rg_df = data_obj.df

    return rg_df


def main(data_obj):

    st.header('Regression')

    rg_df = import_dset(data_obj)

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    rg_method = st.radio(label='Classification Method', options=[
                         'Random Forest', 'Button 1', 'Button 2'])

    # Selected 'Remove outliers'
    if rg_method == 'Random Forest':

        st.dataframe(rg_df)
        st.write(rg_df.shape)




if __name__ == "__main__":
   main()