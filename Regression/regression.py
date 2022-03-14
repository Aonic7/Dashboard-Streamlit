import datetime
import pandas as pd
import numpy as np
import streamlit as st
from .Regression_final import Regressor
from .MLP_Regressor import NN_Regressor, Regressor_Inputs
from .MLP_TimeSeries import NN_TimeSeries_Reg, Regressor_Inputs_TS
from .Regression_Group4 import Regression
from .TimeSeries_Final import Timeseries, rf_Inputs



def main(data_obj):
    """Regression main 

    :param data_obj: DataObject instance.
    :type data_obj: __main__.DataObject
    
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
                                                             'Other Methods'])




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
                    selected_column = st.selectbox("Column to regress:", columns_list)
                    col_idx = rg_df.columns.get_loc(selected_column)

                    solver_fun1 = ("lbfgs", "sgd", "adam")
                    selected_solver = st.selectbox("Solver:", solver_fun1)

                    activation_fun1 = ("identity", "logistic", "tanh", "relu")
                    selected_function = st.selectbox("Activation function:", activation_fun1)

                # Input variables/widgets for the 3rd column
                with cc3:
                    number_hl = st.slider('Hidden layers:', 1, 5, 2, 1)

                    a = [] #

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
                    
                    If you don't group data then timeseries regression will be done on the whole dataset.
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

            # Section header
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)

            # User input widgets/variables
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

            # Submit button
            with st.form(key="form"):
                submit_button = st.form_submit_button(label='Submit')

                # Submit button execution
                if submit_button:
                    # If TimeSeries flag is checked
                    if t_s:
                        try:
                            # Spinner due to a long processing of TimeSeries
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

                        except:
                            st.error("Something went wrong...")

                    # Checkbox is not activated (False value)
                    if t_s == False:
                        try:
                            # Spinner animation
                            with st.spinner("Training models..."):
                                X = rg_df[x_list]
                                Y = rg_df[selected_column]
                                reg_inst = Regressor(X,Y)
                                reg_inst.model(tree_size, tt_proportion)

                            reg_inst.result(reg_inst.Y_test, reg_inst.Y_pred)
                            reg_inst.prediction_plot(reg_inst.Y_test, reg_inst.Y_pred)

                        except:
                            st.error("Something went wrong...")

    ################## Group 4

    if rg_method == 'Other Methods':
        """_summary_
        Args:
            data_obj (_type_): _description_
        """

        # Displaying the Dataframe
        st.dataframe(rg_df)
        # Displaying the shape of the Dataframe
        st.write(rg_df.shape)
        # Button for downloading the Dataframe
        st.download_button(label="Download data as CSV",
                    data=rg_df.to_csv(index=False),
                    file_name='Preprocessed Dataset.csv',
                    mime='text/csv')

        # creating a copy of the current dataframe
        rg_df = data_obj.df.copy()
        # Using this Dataframe to create an instance of the Regression class
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


        with st.container():
            # Creating three columns for Data Description
            c1, c2, c3 = st.columns(3)

            # Left Column shows the data types of every column
            with c1:
                st.subheader("Dataframe's datatypes")
                st.dataframe(rg_df.dtypes.astype(str))

            # Middle Column shows the correlation heatmap (the correlation matrix shows a "stair" structure)
            with c2:
                st.subheader("Correlation heatmap")
                regg_obj.plot_heatmap_correlation()
            # Right Column shows a description of the internal stored Dataframe (using the pd.Dataframe.describe() function)
            with c3:
                st.subheader("Dataframe description")
                st.dataframe(regg_obj.get_dataframe_description())

        # Creating the second row of the page
        with st.container():
            st.subheader('Select input settings')
            # Creating three columns for the Model Inputs
            cc1, cc2, cc3 = st.columns(3)

            # In the left column are input option for the preparation of the dataset
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                del_dup = st.checkbox('Deleting duplicates?')
                scale = st.checkbox('Scale data?')
                del_na = st.checkbox('Get rid of N/A values?')

            # In the middel column is the chosen target column
            with cc2:
                columns_list = list(rg_df.columns)
                selected_column = st.selectbox("Column to regress:", columns_list)

            # In the right column is the selected regression method
            with cc3:
                regression_list = ["Support Vector Machine Regression", "Elastic Net Regression","Ridge Regression",
                                "Linear Regression","Stochastic Gradient Descent Regression"]
                selected_regressor = st.selectbox("Select a regression method:", regression_list)
                # Depending on the selected regression method are different options to setup
                # Following options are given for the Support Vector Machine:
                if selected_regressor == "Support Vector Machine Regression":
                    # Kernel to be selected
                    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid', 'precomputed']
                    selected_kernel = st.selectbox("Kernel:", kernel_list)
                    # Degree (Degree of the polynomial kernel function) to be selected
                    degree_default = 3
                    degree_value = st.number_input('Degree of the polynomial kernel function', 1, 10, degree_default, 1)
                    # svmNumber to be selected (An upper bound on the fraction of training errors and a lower bound of the fraction of
                    # support vectors, should be in the interval (0, 1), defaults to 0.5)
                    svmNumber_default = 0.5
                    svmNumber_value = st.slider('SVM Number', 0.0, 1.0, svmNumber_default, 0.1)
                    # Maximal number of Iterations to be selected
                    # if -1 = no limit
                    maxIterations_default = -1
                    maxIterations_value = st.number_input('Maximum of Iterations', -1, 1000, maxIterations_default, 1)

                # Optional regression method
                if selected_regressor == "Elastic Net Regression":
                    pass

                # Following options are given for the Support Vector Machine:
                if selected_regressor == "Ridge Regression":
                    # Solver to be selected
                    # defaults to 'auto'
                    solver_list = ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga', 'lbfgs']
                    selected_solver = st.selectbox("Solver:", solver_list)

                    # Maximum number of iterations for conjugate gradient solver, defaults to 15000
                    maxIterations_default = 15000
                    maxIterations_value = st.number_input('Maximum of Iterations', 1, 50000, maxIterations_default, 100)

                # Optional regression method
                if selected_regressor == "Linear Regression":
                        pass

                # Following options are given for the Stochastic Gradient Descent Regression:
                if selected_regressor == "Stochastic Gradient Descent Regression":
                    # The maximum number of passes over the training data, defaults to 1000
                    maxIterations_default = 1000
                    maxIterations_value = st.number_input('Maximum of Iterations', 1, 10000, maxIterations_default, 100)


        with st.container():
            # The last row
            cc1, cc2, cc3= st.columns([1,2,2])
            # Scaling the dataframe and storing it in a separated Datframe for later usage
            rg_df_norm = (rg_df - np.min(rg_df)) / (np.max(rg_df) - np.min(rg_df))
            regg_obj_norm = Regression(rg_df_norm)

            with cc1:
                st.subheader('Model Training')
                submit_button = st.button(label='Train Model')

                if submit_button:
                    with st.spinner("Training models..."):
                        try:
                            if selected_regressor == "Support Vector Machine Regression":
                                ################# Support Vector Machine Regression with the internal Dataset
                                # Splitting the internal Dataset in a train and test portion
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                # Building the Support Vector Machine Regression with the internal Dataset
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Support Vector Machine Regression ",
                                                        kernel=selected_kernel,
                                                        degree=degree_value,
                                                        svmNumber=svmNumber_value,
                                                        maxIterations=maxIterations_value)
                                # Outputting the regression plot and regression metrics
                                st.session_state.fig = regg_obj.plot_regression_1()
                                #################
                                # Splitting the scaled Dataset in a train and test portion
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                # Building the Support Vector Machine Regression with the scaled Dataset
                                regg_obj_norm.build_regression("Support Vector Machine Regression ",
                                                        kernel=selected_kernel,
                                                        degree=degree_value,
                                                        svmNumber=svmNumber_value,
                                                        maxIterations=maxIterations_value)
                                # Outputting the Sensitivity plot on the scaled Dataset
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Elastic Net Regression":
                                ###################### Elastic Net Regression with the internal Dataset
                                # Splitting the internal Dataset in a train and test portion
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                # Building the Elastic Net Regression with the internal Dataset
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Elastic Net Regression ")
                                # Outputting the regression plot and regression metrics
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                regg_obj_norm.build_regression("Elastic Net Regression ")
                                # Outputting the Sensitivity plot on the scaled Dataset
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Ridge Regression":
                                ###################### Ridge Regression with the internal Dataset
                                # Splitting the internal Dataset in a train and test portion
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                # Building the Ridge Regression with the internal Dataset
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Ridge Regression ",
                                                        max_iter=maxIterations_value,
                                                        solver=selected_solver)
                                # Outputting the regression plot and regression metrics
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                # Splitting the scaled Dataset in a train and test portion
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                # Building the Ridge Regression with the scaled Dataset
                                regg_obj_norm.build_regression("Ridge Regression ",
                                                        max_iter=maxIterations_value,
                                                        solver=selected_solver)
                                # Outputting the Sensitivity plot on the scaled Dataset
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Linear Regression":
                                ###################### Linear Regression with the internal Dataset
                                # Splitting the internal Dataset in a train and test portion
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                # Building the Linear Regression with the internal Dataset
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Linear Regression ")
                                # Outputting the regression plot and regression metrics
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ###################### Linear Regression with the scaled Dataset
                                # Splitting the scaled Dataset in a train and test portion
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                # Building the Linear Regression with the scaled Dataset
                                regg_obj_norm.build_regression("Linear Regression ")
                                # Outputting the Sensitivity plot on the scaled Dataset
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()

                            if selected_regressor == "Stochastic Gradient Descent Regression":
                                ###################### Stochastic Gradient Descent Regression with the internal Dataset
                                # Splitting the internal Dataset in a train and test portion
                                regg_obj.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=scale,
                                                        deleting_duplicates=del_dup)
                                # Building the Stochastic Gradient Descent Regression with the internal Dataset
                                st.session_state.model,st.session_state.model_string = regg_obj.build_regression("Stochastic Gradient Descent Regression ",
                                                        max_iter=maxIterations_value)
                                # Outputting the regression plot and regression metrics
                                st.session_state.fig = regg_obj.plot_regression_1()
                                ######################
                                # Splitting the scaled Dataset in a train and test portion
                                regg_obj_norm.split_train_test(label_target=selected_column,
                                                        testsize=tt_proportion,
                                                        random_state=0,
                                                        deleting_na=del_na,
                                                        scaling=False,
                                                        deleting_duplicates=del_dup)
                                # Building the Stochastic Gradient Descent Regression with the scaled Dataset
                                regg_obj_norm.build_regression("Stochastic Gradient Descent Regression ",
                                                        max_iter=maxIterations_value)
                                # Outputting the Sensitivity plot on the scaled Dataset
                                st.session_state.fig_norm = regg_obj_norm.MainEffectsPlot()


                        except ValueError as e:
                            st.error("Please check if you selected a dataset and column suitable for the regression "
                                     "model.\n Remember that the regression model only works with numerical data")
                try:
                    # Outputting the Regression Methods metrics
                    st.metric(st.session_state.model_string[0]+str(" --> RMSE"),st.session_state.model_string[1])
                    st.metric(st.session_state.model_string[0]+str(" --> R2-Score"),st.session_state.model_string[2])
                except AttributeError as e:
                    pass

            with cc2:
                st.subheader('Model Graphs')
                try:
                    # Plotting the Regresssion Plot
                    st.caption("Actual- vs Expected Target Value")
                    st.pyplot(st.session_state.fig)
                    # Plotting the Sensitivity Plot
                    st.caption("Main Effects Plot")
                    st.pyplot(st.session_state.fig_norm)
                except:
                    pass

            with cc3:
                # This Column creates the Ability for the User
                # to use the trained model to make a prediction
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