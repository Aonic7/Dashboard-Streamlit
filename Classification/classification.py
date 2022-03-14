# General import section 
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
from sklearn.preprocessing import MinMaxScaler #for data preparation
from sklearn.model_selection import train_test_split #for data split
from math import isqrt #for input variable calculation

# Local classes/functions import
from .MLP_Classifier import NN_Classifier, classifier_inputs
from .ClassificationClass import Classification
from .RF_classfication import Sample



def main(data_obj):
    """Classification main 

    :param data_obj: DataObject instance.
    :type data_obj: __main__.DataObject
    
    """

    with st.expander("How to use", expanded=True):
        st.markdown("""
                    Here the User can choose Classification method and parameters for each different classifier:
                    1. Choose Classifier Type (Neural Network, Random Forest, SVM, …etc)
                    2. Choose Label column in the data set
                    3. Modify Chosen Classifier inputs and click submit to run the classification and output results
                    """)

    # Page header
    st.header("Classification")

    # Check for existing dataset 
    try:
        var_read = pd.read_csv("Smoothing_and_Filtering//Preprocessing dataset.csv")
        cl_df = var_read

    except:
        cl_df = data_obj.df
        st.error("""You did not smooth of filter the data.
                     Please go to 'Smoothing and filtering' and finalize your results.
                     Otherwise, the default dataset would be used!
                     """)   
    
    # Radio button style/setting
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    cl_method = st.radio(label='Classification Method', options=[
                         'Neural Networks', 'Random Forest', 'Other Methods'])

    # Show dataset
    st.dataframe(cl_df)
    st.write(cl_df.shape)
    st.download_button(label="Download data as CSV",
                    data=cl_df.to_csv(index=False),
                    file_name='Preprocessed Dataset.csv',
                    mime='text/csv',
                                                )

    # Selected 'Neural Networks'
    if cl_method == 'Neural Networks':

        with st.container():

            # Input settings header
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            # Input variables/widgets for the 1st column
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                iteration_num = st.slider('Number of iterations', 100, 5000, 200, 50)
                norm_bool = st.checkbox('Normalize data?')
                resample_bool = st.checkbox('Resample data?')              
                
            # Input variables/widgets for the 2nd column
            with cc2:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)
                col_idx = cl_df.columns.get_loc(selected_column)

                solver_fun1 = ("lbfgs", "sgd", "adam")
                selected_solver = st.selectbox("Solver:", solver_fun1)

                activation_fun1 = ("identity", "logistic", "tanh", "relu")
                selected_function = st.selectbox("Activation function:", activation_fun1)           
            
            # Input variables/widgets for the 3rd column
            with cc3:
                number_hl = st.slider('Hidden layers:', 1, 5, 2, 1)

                a = []

                for i in range(number_hl):
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 100, 1, 1, key=i))
        
        with st.container():
            
            # Submit button
            with st.form(key="Youssef"):
                submit_button = st.form_submit_button(label='Submit')

                # Circle animation for code execution 
                if submit_button:
                    with st.spinner("Training models..."):
                        
                        # Class instance for further input
                        NN_inputs = classifier_inputs(tt_proportion,
                                                selected_function,
                                                tuple(a),
                                                selected_solver,
                                                iteration_num,
                                                norm_bool,
                                                resample_bool
                                                )

                        # Class instance/method for Neural Networks execution
                        Classifier = NN_Classifier(cl_df, NN_inputs, col_idx)

                        Classifier.Classify()
                        Classifier.printing()
                        Classifier.Conf()


    # Selected 'Classification (Noah)'
    if cl_method == 'Other Methods':
       
        st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>',
        unsafe_allow_html=True)

    # Main data classification method radio selector
        # Creating an Instance of the Classification Class with the Classification Object
        class_obj = Classification(cl_df)
        
        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            # In the left column are input option for the preparation of the dataset
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                upsample_cl = st.checkbox('Upsample data?')
                scale = st.checkbox('Scale data?')
                del_na = st.checkbox('Get rid of N/A values?')
            # In the middel column is the chosen target column of the dataset
            with cc2:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)

            # In the right column is the selected classifier
            with cc3:
                classifier_list = ["KNN", "SVM", "LR"]
                selected_classifier = st.selectbox("Select a classifier:", classifier_list)
                # depending on the selected classifier are different options to setup
                # Following options are given for the K-Nearest Neighbor
                if selected_classifier == "KNN":
                    k_default = isqrt(cl_df.shape[0]) # Default K-Value is the square root of the number of sampels
                    if k_default >= 200: k_default = 200
                    k_value = st.number_input('"k" value:', 1, 200, k_default, 1)
                # Following options are given for the Support Vector Machine
                if selected_classifier == "SVM":
                    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid'] # default kernel is "linear"
                    selected_kernel = st.selectbox("Kernel:", kernel_list)
                # Following options are given for the Logistic Regression
                if selected_classifier == "LR":
                    solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']  # default solver is "liblinear"
                    selected_solver = st.selectbox("Solver:", solver_list)

        with st.container():
            cc1, cc2, cc3 = st.columns(3)

            with cc1:
                st.subheader('Model Training')
                submit_button = st.button(label='Train Model')

                if submit_button:
                    with st.spinner("Training models..."):

                        try:
                            if selected_classifier == "KNN":
                                # K Nearest Neighbor
                                # Splitting the Dataset in the Train and Test portion
                                class_obj.split_train_test(y_column_name=selected_column,
                                                        test_size=tt_proportion,
                                                        random_state=0,
                                                        upsample=upsample_cl,
                                                        scaling=scale,
                                                        deleting_na=del_na)
                                # Building the KNN Classifier
                                st.session_state.model, \
                                st.session_state.pred =class_obj.build_classifier('KNN', k_value)
                                # Generating the Plot and Metrics of the Classifier model
                                st.session_state.fig_class,\
                                st.session_state.cnf_matrix,\
                                st.session_state.report = class_obj.show_classifier_accuracy()

                            if selected_classifier == "SVM":
                                # Support Vector Machine
                                # Splitting the Dataset in the Train and Test portion
                                class_obj.split_train_test(y_column_name=selected_column,
                                                        test_size=tt_proportion,
                                                        random_state=0,
                                                        upsample=upsample_cl,
                                                        scaling=scale,
                                                        deleting_na=del_na)
                                # Building the SVM Classifier
                                st.session_state.model, \
                                st.session_state.pred =class_obj.build_classifier('SVM', selected_kernel)
                                # Generating the Plot and Metrics of the Classifier model
                                st.session_state.fig_class,\
                                st.session_state.cnf_matrix,\
                                st.session_state.report = class_obj.show_classifier_accuracy()

                            if selected_classifier == "LR":
                                # Logistic Regression
                                # Splitting the Dataset in the Train and Test portion
                                class_obj.split_train_test(y_column_name=selected_column,
                                                        test_size=tt_proportion,
                                                        random_state=0,
                                                        upsample=upsample_cl,
                                                        scaling=scale,
                                                        deleting_na=del_na)
                                # Building the LR Classifier
                                st.session_state.model, \
                                st.session_state.pred =class_obj.build_classifier('LR', selected_solver)
                                # Generating the Plot and Metrics of the Classifier model
                                st.session_state.fig_class,\
                                st.session_state.cnf_matrix,\
                                st.session_state.report = class_obj.show_classifier_accuracy()

                        except ValueError as e:
                            st.error(
                                "Please check if you selected a dataset and column suitable for binary classification. \nAlternatively, your labels should be one-hot encoded.")
                try:
                    # Printing the Overall Accuracy
                    st.metric("Overall Accuracy of "+str(st.session_state.model),st.session_state.pred)
                    for dim in range(st.session_state.cnf_matrix.shape[0]):
                        # Printing the accuracy for every class
                        accuracy = st.session_state.cnf_matrix[dim][dim] / sum(st.session_state.cnf_matrix[dim])
                        # st.write("Accuracy for the ",dim,"class:", accuracy)
                        # st.write(f"Accuracy for the {dim} class:")
                        st.metric(f"Accuracy for the {dim} class:", round(accuracy, 4))
                except:
                    pass

            with cc2:
                st.subheader('Model Graphs')
                try:
                    # Plotting the Confusion Matrix
                    st.caption("Confusion Matrix")
                    st.pyplot(st.session_state.fig_class)
                    # Plotting the Classification Report
                    st.caption("Classification Report")
                    st.dataframe(st.session_state.report)
                except:
                    pass

            with cc3:
                # This Column creates the Ability for the User
                # to use the trained model to make a prediction
                st.subheader('Model Prediction')
                columns_list = list(cl_df.columns)
                parameter = []

                for i , column in enumerate(columns_list):
                    if not column == selected_column:
                        parameter.append(st.number_input(label=column,
                                                        min_value=0,
                                                        max_value=10000,
                                                        value=10,
                                                        step=10))
                predict_button = st.button(label='Predict')
                if predict_button:
                    try:
                        prediction = st.session_state.model.predict(pd.DataFrame([parameter]))
                        st.metric("Prediction of "+selected_column, round(prediction[0], 4))
                    except:
                        st.write("No model found!")
        
    # Selected 'Random Forest (Sneha)'
    if cl_method == 'Random Forest':

        # Container for inputs
        with st.container():

            # Input section subheader
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            # Input variables/widgets for the 1st column
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)

            # Input variables/widgets for the 2nd column
            with cc2:
                estimator_value = st.slider('Estimator:', 0, 1000, 500, 10)

            # Input variables/widgets for the 3rd column
            with cc3:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)

            # Container for output
            with st.container():

                # Section subheader
                st.subheader("Model")


                # Submit button
                with st.form(key="Sneha"):
                    submit_button = st.form_submit_button(label='Submit')

                    
                    if submit_button:
                        # Circle animation for code execution
                        with st.spinner("Training models..."):
                            
                            # Instance method call
                            try:   
                                # Input data preparation
                                scaler = MinMaxScaler()
                                features = [s for s in cl_df.columns if s != selected_column]
                                
                                scaled_Dataframe = pd.DataFrame(data = cl_df)
                                scaled_Dataframe[features] = scaler.fit_transform(cl_df[features])

                                X = scaled_Dataframe[features]  # contains the features
                                Y = scaled_Dataframe[selected_column]  # contains the target column 
                                
                                # Instance creation
                                obj = Sample(X, Y)

                                obj.model(estimator_value, tt_proportion) 
                                obj.report()
                                obj.accuracy()

                            except ValueError as e:
                                st.error('Something went wrong, Sneha...')
        
# Main
if __name__ == "__main__":
    main()
