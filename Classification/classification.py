from ast import Not
from code import interact
from typing import Collection
from collections import namedtuple
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype
from .MLP_Classifier import NN_Classifier, classifier_inputs

from sklearn.metrics import classification_report


def import_dset(data_obj):
    try:
        cl_df = pd.read_csv(
            'Smoothing_and_Filtering//Preprocessing Dataset.csv', index_col=None)

    except:
        st.error("""You did not smooth of filter the data.
                    Please go to 'Smoothing and filtering' and finalize your results.
                    Otherwise, the default dataset would be used!
                    """)
        cl_df = data_obj.df

    return cl_df


def main(data_obj):

    st.header("Classification")

    cl_df = import_dset(data_obj)

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    cl_method = st.radio(label='Classification Method', options=[
                         'Neural Networks', 'Button 1', 'Button 2'])

    # Selected 'Remove outliers'
    if cl_method == 'Neural Networks':

        st.dataframe(cl_df)
        st.write(cl_df.shape)

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)

                iteration_num = st.slider('Number of iterations', 100, 500, 200, 50)

                norm_bool = st.select_slider('Normalize data?', [False, True], False)              
                
            
            with cc2:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)
                col_idx = cl_df.columns.get_loc(selected_column)

                solver_fun1 = ("lbfgs", "sgd", "adam")
                selected_solver = st.selectbox("Solver:", solver_fun1)

                activation_fun1 = ("identity", "logistic", "tanh", "relu")
                selected_function = st.selectbox("Activation function:", activation_fun1)           
            
            with cc3:
                number_hl = st.slider('Hidden layers:', 1, 5, 2, 1)

                a = []

                for i in range(number_hl):
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 10, 1, 1, key=i))


               
        with st.container():
            NN_inputs = classifier_inputs(tt_proportion,
                                       selected_function,
                                       tuple(a),
                                       selected_solver,
                                       iteration_num,
                                       norm_bool
                                       )

            Classifier = NN_Classifier(cl_df, NN_inputs, col_idx)

            # st.write(Classifier.NN_Outputs.Report)
            Classifier.Classify()
            Classifier.printing()
            # Classifier.Conf()
            # st.write(classification_report(getattr(Classifier, 'NN_Outputs.NN_Inputs')))
                 

if __name__ == "__main__":
    main()
