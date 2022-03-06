from ast import Not
from code import interact
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype
from .Regression_final import Regressor
from .MLP_Regressor import NN_Regressor,Regressor_Inputs

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
                         'Random Forest', 'Neural Network', 'Button 2'])

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

    if rg_method == 'Neural Network':

        st.dataframe(rg_df)
        st.write(rg_df.shape)

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)

                iteration_num = st.slider('Number of iterations', 100, 500, 200, 50)

                #norm_bool = st.select_slider('Normalize data?', [False, True], False)              
                
            
            with cc2:
                columns_list = list(rg_df.columns)
                selected_column = st.selectbox("Column to Predict:", columns_list)
                col_idx = rg_df.columns.get_loc(selected_column)

                solver_fun1 = ("lbfgs", "sgd", "adam")
                selected_solver = st.selectbox("Solver:", solver_fun1)

                activation_fun1 = ("identity", "logistic", "tanh", "relu")
                selected_function = st.selectbox("Activation function:", activation_fun1)           
            
            with cc3:
                number_hl = st.slider('Hidden layers:', 1, 5, 2, 1)

                a = []

                for i in range(number_hl):
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 20, 1, 1, key=i))

        with st.container():
            NN_inputs = Regressor_Inputs(tt_proportion,
                                       selected_function,
                                       tuple(a),
                                       selected_solver,
                                       iteration_num
                                       #norm_bool
                                       )
            st.write(type(tt_proportion))

            MLP_Regressor = NN_Regressor(rg_df, NN_inputs, col_idx)

            MLP_Regressor.Regressor()
            # st.write(Classifier.Classify()[52])
            MLP_Regressor.printing()
            
            MLP_Regressor.plotting()

if __name__ == "__main__":
   main()