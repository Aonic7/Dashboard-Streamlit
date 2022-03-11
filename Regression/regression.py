import datetime
from code import interact
import pandas as pd
import numpy as np
import streamlit as st
from pandas.api.types import is_numeric_dtype
from .Regression_final import Regressor
from .MLP_Regressor import NN_Regressor, Regressor_Inputs
from .MLP_TimeSeries import NN_TimeSeries_Reg, Regressor_Inputs_TS
from .RegressionClass import Regression
from .TimeSeries_Final import Timeseries, rf_Inputs

#dateparse = lambda x: datetime.strptime(x, '%Y-%m-%d %H:%M:%S')



def main(data_obj):
    """_summary_

    Args:
        data_obj (_type_): _description_
    """ 

    st.header('Regression')

    with st.expander("How to use", expanded=True):
        st.markdown("""
                    Here the User can choose Regression Method and parameters for each different regressor:
                    1. Choose Regressor type and weather it is time series regression or not?
                    2. Choose Target Column for the regression
                    3. Modify Regressor inputs and click submit to run regression and output results
                    """)

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
    rg_method = st.radio(label='Regression Method', options=['Neural Networks',
                                                             'Random Forest',
                                                             'Other Method'])

    

    
    # Selected 'Neural Networks'
    if rg_method == 'Neural Networks':
        rg_nn_radio = st.radio(label = 'Neural Network',
                             options = ['Standard','Timeseries'])

        st.dataframe(rg_df)
        st.write(rg_df.shape)
        st.download_button(label="Download data as CSV",
                data=rg_df.to_csv(index=False),
                file_name='Preprocessed Dataset.csv',
                mime='text/csv')
        
        if rg_nn_radio == 'Standard':
        
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
        if rg_nn_radio == 'Timeseries':
        
            with st.expander("How to use", expanded=True):
                st.markdown("""
                    Here you need to choose if you want to group the data set based on a unique value in a specific column and then run the regression for this filtered data only. 
                    
                    If you don't group data then timeseries regression will be done on the whole model.
                    """)
                    
            with st.container():

                # Input settings header
                st.subheader('Select input settings')

                cc1, cc2, cc3 = st.columns(3)
                
                
                # Input variables/widgets for the 1st column
                try:
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
                except KeyError as e:
                    st.error("Are you sure this dataset has a time column?")
                    st.stop()


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
    if rg_method == 'Random Forest':

        st.dataframe(rg_df)
        st.write(rg_df.shape)
        st.download_button(label="Download data as CSV",
                    data=rg_df.to_csv(index=False),
                    file_name='Preprocessed Dataset.csv',
                    mime='text/csv')

      
        with st.container():
            st.subheader('Select input settings')

            
            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                columns_list = list(rg_df.columns)

                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.3, 0.05)
                tree_size = st.slider('Number of trees (estimator):', 100, 1000, 100, 100)

                selected_column = st.selectbox("Select target column:", columns_list)
                x_list = [s for s in columns_list if s != selected_column]

                t_s = st.checkbox("Is this a time-series?")

            with cc2:
                if t_s:
                    column_uniques_list = list(rg_df.select_dtypes(exclude=['datetime']).columns)
                    column_uniques = st.selectbox("Select column with unique values:", column_uniques_list)
                    tm_columns_list = list(rg_df.select_dtypes(include=['datetime']).columns)
                    time_column = st.selectbox("Select a time column:", tm_columns_list)
                    base_column_list = [s for s in columns_list if s != selected_column and s != time_column]
                    base_column = st.selectbox("Select base column:", base_column_list)




            with cc3:
                if t_s:
                    unique_val = st.selectbox("Select a unique:", list(rg_df[column_uniques].unique()))
                    d = st.date_input("Date:", datetime.date(2016, 10, 10))
                    t = st.time_input('Time:', datetime.time(15, 15))
                    
       
        with st.container():
                       
            # calling the methods using object 'obj'
            with st.form(key="form"):
                submit_button = st.form_submit_button(label='Submit')

                if submit_button:
                    if t_s:
                        try:
                            with st.spinner("Training models..."):
                                dat = datetime.datetime(d.year,d.month,d.day)
                                dt_input = datetime.datetime(d.year,d.month,d.day,t.hour,t.minute)
                                ts_rg_input = rf_Inputs(unique_val, dat, dt_input)

                                obj= Timeseries(rg_df, ts_rg_input,
                                                base_column, selected_column,
                                                time_column, column_uniques)

                                obj.DataPrep()
                                obj.fullmodel()
                                obj.model(tree_size, tt_proportion)
                                obj.Results()
                                obj.SelectedSysCode(ts_rg_input)
                                obj.UserSelectedmodel(tree_size, tt_proportion)
                        except KeyError as e:
                            st.error("Guess what? You hardcoded it again!")
                        except:
                            st.error("Something went wrong, Sneha...")

                    if t_s == False:        
                        try:
                            with st.spinner("Training models..."):
                                X = rg_df[x_list]
                                Y = rg_df[selected_column]
                                reg_inst = Regressor(X,Y) 
                                reg_inst.model(tree_size, tt_proportion)
                                st.balloons()

                            reg_inst.result(reg_inst.Y_test, reg_inst.Y_pred)
                            reg_inst.prediction_plot(reg_inst.Y_test, reg_inst.Y_pred)
                            
                        except:
                            st.balloons()
                            st.error("Something went wrong, Sneha...")

    if rg_method == 'Other Method':
        """_summary_
        Args:
            data_obj (_type_): _description_
        """

        st.dataframe(rg_df)
        st.write(rg_df.shape)
        st.download_button(label="Download data as CSV",
                    data=rg_df.to_csv(index=False),
                    file_name='Preprocessed Dataset.csv',
                    mime='text/csv')


        st.header('Regression')

        rg_df = data_obj.df.copy()
        regg_obj = Regression(rg_df)

        for col in rg_df.columns:
            if rg_df[col].dtype == 'object':
                try:
                    rg_df[col] = pd.to_datetime(rg_df[col])
                except ValueError:
                    pass

        st.write(
            '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>',
            unsafe_allow_html=True)

        st.dataframe(rg_df)
        st.write(rg_df.shape)

        with st.container():
            c1, c2, c3 = st.columns(3)

            with c1:
                st.subheader("Dataframe's datatypes")
                st.dataframe(rg_df.dtypes.astype(str))
                # st.download_button(label="Download data as CSV",
                #                 data=rg_df.to_csv(index=False),
                #                 file_name='Preprocessed Dataset.csv',
                #                 mime='text/csv',
                #                 )
            with c2:
                st.subheader("Correlation heatmap")
                regg_obj.plot_heatmap_correlation()

            with c3:
                st.subheader("Dataframe description")
                st.dataframe(regg_obj.get_dataframe_description())

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)

            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                del_dup = st.checkbox('Deleting duplicates?')
                scale = st.checkbox('Scale data?')
                del_na = st.checkbox('Get rid of N/A values?')

            with cc2:
                columns_list = list(rg_df.columns)
                selected_column = st.selectbox("Column to regress:", columns_list)

                #columns_list = list(rg_df.columns)
                #selected_drop_column = st.selectbox("Column to drop:", columns_list)
                #if st.button(label='Drop Column'):
                #    rg_df = regg_obj.dropColumns(selected_drop_column)


            with cc3:
                regression_list = ["Support Vector Machine Regression", "Elastic Net Regression","Ridge Regression",
                                "Linear Regression","Stochastic Gradient Descent Regression"]
                selected_regressor = st.selectbox("Select a regression method:", regression_list)

                if selected_regressor == "Support Vector Machine Regression":
                    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                    selected_kernel = st.selectbox("Kernel:", kernel_list)

                    degree_default = 3
                    degree_value = st.number_input('Degree of the polynomial kernel function', 1, 10, degree_default, 1)

                    svmNumber_default = 0.5
                    svmNumber_value = st.slider('SVM Number', 0.0, 1.0, svmNumber_default, 0.1)

                    maxIterations_default = -1
                    maxIterations_value = st.number_input('Maximum of Iterations', -1, 1000, maxIterations_default, 1)

                if selected_regressor == "Elastic Net Regression":
                    pass

                if selected_regressor == "Ridge Regression":
                    solver_list = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
                    selected_solver = st.selectbox("Solver:", solver_list)

                    maxIterations_default = 15000
                    maxIterations_value = st.number_input('Maximum of Iterations', 1, 50000, maxIterations_default, 100)

                if selected_regressor == "Linear Regression":
                        pass

                if selected_regressor == "Stochastic Gradient Descent Regression":
                    maxIterations_default = 1000
                    maxIterations_value = st.number_input('Maximum of Iterations', 1, 10000, maxIterations_default, 100)


        with st.container():
            cc1, cc2, cc3= st.columns([1,2,2])

            rg_df_norm = (rg_df - np.min(rg_df)) / (np.max(rg_df) - np.min(rg_df))
            regg_obj_norm = Regression(rg_df_norm)

            with cc1:
                st.subheader('Model Training')
                submit_button = st.button(label='Train Model')

                if submit_button:
                    with st.spinner("Training models..."):
                        try:
                            if selected_regressor == "Support Vector Machine Regression":
                                #################
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Support Vector Machine Regression ",
                                                        kernel=selected_kernel,
                                                        degree=degree_value,
                                                        svmNumber=svmNumber_value,
                                                        maxIterations=maxIterations_value)
                                st.session_state.fig = regg_obj.plot_regression_1()
                                #################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Support Vector Machine Regression ",
                                                        kernel=selected_kernel,
                                                        degree=degree_value,
                                                        svmNumber=svmNumber_value,
                                                        maxIterations=maxIterations_value)
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Elastic Net Regression":
                                ######################
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Elastic Net Regression ")
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Elastic Net Regression ")
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Ridge Regression":
                                ######################
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Ridge Regression ",
                                                        max_iter=maxIterations_value,
                                                        solver=selected_solver)
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Ridge Regression ",
                                                        max_iter=maxIterations_value,
                                                        solver=selected_solver)
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Linear Regression":
                                ######################
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Linear Regression ")
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Linear Regression ")
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Stochastic Gradient Descent Regression":
                                ######################
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Stochastic Gradient Descent Regression ",
                                                        max_iter=maxIterations_value)
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Stochastic Gradient Descent Regression ",
                                                        max_iter=maxIterations_value)
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()


                        except ValueError as e:
                            st.error(
                                "Please check if you selected a dataset and column suitable for binary classification. \nAlternatively, your labels should be one-hot encoded.")
                try:
                    st.metric(st.session_state.model_string[0]+str(" --> RMSE"),st.session_state.model_string[1])
                    st.metric(st.session_state.model_string[0]+str(" --> R2-Score"),st.session_state.model_string[2])
                except AttributeError as e:
                    pass

            with cc2:
                st.subheader('Model Graphs')
                try:
                    st.caption("Actual- vs Expected Target Value")
                    st.pyplot(st.session_state.fig)
                    st.caption("Main Effects Plot")
                    st.pyplot(st.session_state.fig_norm)
                except:
                    pass

            with cc3:
                st.subheader('Model Prediction')
                columns_list = list(rg_df.columns)
                parameter = []

                for i , column in enumerate(columns_list):
                    if not column == selected_column:
                        parameter.append(st.number_input(label=column,
                                                        min_value=0.0,
                                                        max_value=100.0,
                                                        value=0.0,
                                                        step=1.0))

                submit_button = st.button(label='Predict')
                if submit_button:
                    try:
                        prediction = st.session_state.model.predict(pd.DataFrame([parameter]))
                        st.metric("Prediction of "+selected_column,round(prediction[0],4))
                    except:
                        st.write("No model found!")


            

if __name__ == "__main__":
   main()