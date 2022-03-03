from ast import Not
from code import interact
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype
from .Regression_final import Regressor

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
    rg_method = st.radio(label='Regression Method', options=[
                         'Random Forest', 'Button 1', 'Button 2'])

    # Selected 'Remove outliers'
    if rg_method == 'Random Forest':

        st.dataframe(rg_df)
        st.write(rg_df.shape)

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.3, 0.05)

            with cc2:
                tree_size = st.slider('Tree size', 100, 1000, 100, 100)
            
          
        with st.container():
            # Iy PF	e dIf If
            # st.write(rg_df.columns)
            X= rg_df[['Iy', 'PF', 'e', 'dIf']]
            Y= rg_df['If']

            reg_inst = Regressor(X,Y)

            # # calling the methods using object 'obj'
            reg_inst.model(tree_size, tt_proportion)
            reg_inst.result(reg_inst.Y_test, reg_inst.Y_pred)
            reg_inst.prediction_plot(reg_inst.Y_test, reg_inst.Y_pred)

if __name__ == "__main__":
   main()