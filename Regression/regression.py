import datetime
from code import interact
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype
from .Regression_final import Regressor
from .MLP_Regressor import NN_Regressor, Regressor_Inputs
from .MLP_TimeSeries import NN_TimeSeries_Reg, Regressor_Inputs_TS

#dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



def main(data_obj):
    """_summary_

    Args:
        data_obj (_type_): _description_
    """

    st.header('Regression')

    try:
        var_read = pd.read_csv("Smoothing_and_Filtering//Preprocessing dataset.csv", index_col=None, parse_dates=True, date_parser = pd.to_datetime)
        rg_df = var_read
        for col in rg_df.columns:
            if rg_df[col].dtype == 'object':
                try:
                    rg_df[col] = pd.to_datetime(rg_df[col])
                except ValueError:
                    pass

    except:
        rg_df = data_obj.df.copy()
        for col in rg_df.columns:
            if rg_df[col].dtype == 'object':
                try:
                    rg_df[col] = pd.to_datetime(rg_df[col])
                except ValueError:
                    pass

        st.error("""You did not smooth of filter the data.
                     Please go to 'Smoothing and filtering' and finalize your results.
                     Otherwise, the default dataset would be used!
                     """)   

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    rg_method = st.radio(label='Regression Method', options=['Neural Networks (Youssef)', 'Neural Networks TS (Youssef)',
                                                             'Richard', 'Richard TS',
                                                             'Random Forest (Sneha)', 'Random Forest TS (Sneha)'])

    st.dataframe(rg_df)
    st.write(rg_df.shape)
    # st.dataframe(rg_df.dtypes.astype(str))
    st.download_button(label="Download data as CSV",
                data=rg_df.to_csv(index=False),
                file_name='Preprocessed Dataset.csv',
                mime='text/csv',
                                            )

    
    # Selected 'Neural Networks'
    if rg_method == 'Neural Networks (Youssef)':
        
        with st.container():

            # Input settings header
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            # Input variables/widgets for the 1st column
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                iteration_num = st.slider('Number of iterations', 100, 5000, 200, 50)
                norm_bool = st.checkbox('Normalize data?')           
                
            # Input variables/widgets for the 2nd column
            with cc2:
                columns_list = list(rg_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)
                col_idx = rg_df.columns.get_loc(selected_column)

                solver_fun1 = ("lbfgs", "sgd", "adam")
                selected_solver = st.selectbox("Solver:", solver_fun1)

                activation_fun1 = ("identity", "logistic", "tanh", "relu")
                selected_function = st.selectbox("Activation function:", activation_fun1)           
            
            # Input variables/widgets for the 3rd column
            with cc3:
                number_hl = st.slider('Hidden layers:', 1, 5, 2, 1)

                a = []

                for i in range(number_hl):
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 600, 1, 1, key=i))
        
        with st.container():
            
            # Submit button
            with st.form(key="Youssef"):
                submit_button = st.form_submit_button(label='Submit')

                # Circle animation for code execution 
                if submit_button:
                    with st.spinner("Training models..."):
                        
                        # Class instance for further input
                        NN_inputs = Regressor_Inputs(tt_proportion,
                                                selected_function,
                                                tuple(a),
                                                selected_solver,
                                                iteration_num,
                                                norm_bool
                                                )

                        # Class instance/method for Neural Networks execution
                        RegressorMLP = NN_Regressor(rg_df, NN_inputs, col_idx)

                        RegressorMLP.Regressor()
                        RegressorMLP.printing()
                        RegressorMLP.plotting()


    # Selected 'Neural Networks TS (Youssef)'             
    if rg_method == 'Neural Networks TS (Youssef)':
    
        with st.container():

            # Input settings header
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            # Input variables/widgets for the 1st column
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                
                solver_fun1 = ("lbfgs", "sgd", "adam")
                selected_solver = st.selectbox("Solver:", solver_fun1)

                activation_fun1 = ("identity", "logistic", "tanh", "relu")
                selected_function = st.selectbox("Activation function:", activation_fun1) 

                group_bool = st.checkbox('Group data?')
                                         
            # Input variables/widgets for the 2nd column
            with cc2:
                iteration_num = st.slider('Number of iterations', 100, 5000, 200, 50)

                columns_list = list(rg_df.select_dtypes(exclude=['object', 'datetime']).columns)
                selected_column = st.selectbox("Column to regress:", columns_list)
                col_idx = rg_df.columns.get_loc(selected_column)

                unique_columns_list = list(rg_df.select_dtypes(exclude=['datetime']).columns)
                unique_selected_column = st.selectbox("Filter uniques:", unique_columns_list)
                unique_col_idx = rg_df.columns.get_loc(unique_selected_column)

                tm_columns_list = list(rg_df.select_dtypes(include=['datetime']).columns)
                time_column = st.selectbox("Select a time column:", tm_columns_list)
                tm_col_idx = rg_df.columns.get_loc(time_column)

                            
            # Input variables/widgets for the 3rd column
            with cc3:
                number_hl = st.slider('Hidden layers:', 1, 5, 3, 1)

                a = []

                for i in range(number_hl):
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 600, 10, 1, key=i))
        
        with st.container():
            
            

            # Class instance for further input
            NN_inputs_TS = Regressor_Inputs_TS(tt_proportion,
                                    selected_function,
                                    tuple(a),
                                    selected_solver,
                                    iteration_num,
                                    group_bool
                                    )

            # Class instance/method for Neural Networks execution
            Regressor_TS = NN_TimeSeries_Reg(rg_df, NN_inputs_TS, col_idx, tm_col_idx)

            if group_bool:
                # Section subheader
                st.subheader('Additional user inputs')
                Regressor_TS.listing(unique_col_idx)
                #st.dataframe(Regressor_TS.group_object)
                selected_group = Regressor_TS.group_object['index']
                sel = st.selectbox("Select an element for groupping:", selected_group)
                sel_idx = selected_group[selected_group == sel].index[0]

            # Submit button
            with st.form(key="Youssef"):
                submit_button = st.form_submit_button(label='Submit')

                

                # Circle animation for code execution 
                if submit_button:
                    with st.spinner("Training models..."):
                        
                        if group_bool:
                            Regressor_TS.group(sel_idx)
                        Regressor_TS.Regressor()
                        Regressor_TS.printing()
                        Regressor_TS.plotting()
                

    # Selected 'Random Forest'
    if rg_method == 'Random Forest (Sneha)':

        # st.dataframe(rg_df)
        # st.write(rg_df.shape)

        
        with st.container():
            st.subheader('Select input settings')

            
            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.3, 0.05)
                tree_size = st.slider('Tree size', 100, 1000, 100, 100)
                t_s = st.checkbox("Is this a time-series?")

            with cc2:
                columns_list = list(rg_df.columns)
                if t_s:
                    column_uniques = st.selectbox("Select column with unique values:", columns_list)
                    tm_columns_list = list(rg_df.select_dtypes(include=['datetime']).columns)
                    time_column = st.selectbox("Select a time column:", tm_columns_list)

                selected_column = st.selectbox("Select label column:", columns_list)
                # x_list = columns_list.remove(selected_column)
                x_list = [s for s in columns_list if s != selected_column]


            with cc3:
                if t_s:
                    unique_val = st.selectbox("Select a unique:", list(rg_df[column_uniques].unique()))
                    d = st.date_input("Date:", datetime.date(2022, 2, 24))
                    t = st.time_input('Time:', datetime.time(13, 37))

       
        with st.container():
            X= rg_df[x_list]
            Y= rg_df[selected_column]

            reg_inst = Regressor(X,Y)

            # calling the methods using object 'obj'
            with st.form(key="form"):
                submit_button = st.form_submit_button(label='Submit')

                if submit_button:

                    cc11, cc22, cc33 = st.columns(3)
                    with cc11:
                        with st.spinner("Training models..."):
                            reg_inst.model(tree_size, tt_proportion)

                    reg_inst.result(reg_inst.Y_test, reg_inst.Y_pred)
                    reg_inst.prediction_plot(reg_inst.Y_test, reg_inst.Y_pred)


            

if __name__ == "__main__":
   main()