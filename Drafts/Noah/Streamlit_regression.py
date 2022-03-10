import datetime
import pandas as pd
import streamlit as st
from RegressionClass import Regression
import numpy as np
import pickle

def import_dset(data_obj):
    try:
        st.write('Try execution')
        rg_df = pd.read_csv('Smoothing_and_Filtering//Preprocessing Dataset.csv', index_col=None)

    except:
        st.write('Exception execution')
        cl_df1 = data_obj.df
        st.error("""You did not smooth of filter the data.
                    Please go to 'Smoothing and filtering' and finalize your results.
                    Otherwise, the default dataset would be used!
                    """)
    return rg_df

def main(data_obj):
    """_summary_
    Args:
        data_obj (_type_): _description_
    """

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
            st.download_button(label="Download data as CSV",
                               data=rg_df.to_csv(index=False),
                               file_name='Preprocessed Dataset.csv',
                               mime='text/csv',
                               )
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

            st.metric(st.session_state.model_string[0]+str(" --> RMSE"),st.session_state.model_string[1])
            st.metric(st.session_state.model_string[0]+str(" --> R2-Score"),st.session_state.model_string[2])

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