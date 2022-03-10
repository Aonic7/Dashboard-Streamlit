#General import section
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from math import isqrt

#Local classes/functions import
from .MLP_Classifier import NN_Classifier, classifier_inputs
from .ClassificationClass_Noah import Classification
from .RF_classfication import Sample


def main(data_obj):
    """Main callable 

    Args:
        data_obj (__main__.DataObject): DataObject instance.
    """

    #Page header
    st.header("Classification")

    #Check for existing dataset 
    try:
        var_read = pd.read_csv("Smoothing_and_Filtering//Preprocessing dataset.csv")
        cl_df = var_read

    except:
        cl_df = data_obj.df
        st.error("""You did not smooth of filter the data.
                     Please go to 'Smoothing and filtering' and finalize your results.
                     Otherwise, the default dataset would be used!
                     """)   
    
    #Radio button style/setting
    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    cl_method = st.radio(label='Classification Method', options=[
                         'Neural Networks (Youssef)', 'Classification (Noah)', 'Random Forest (Sneha)'])

    #Show dataset
    st.dataframe(cl_df)
    st.write(cl_df.shape)
    st.download_button(label="Download data as CSV",
                    data=cl_df.to_csv(index=False),
                    file_name='Preprocessed Dataset.csv',
                    mime='text/csv',
                                                )

    # Selected 'Neural Networks'
    if cl_method == 'Neural Networks (Youssef)':

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)

                iteration_num = st.slider('Number of iterations', 100, 5000, 200, 50)
                norm_bool = st.checkbox('Normalize data?')
                bool_placeholder = st.checkbox('BOOLEAN PLACEHOLDER')              
                
            
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
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 100, 1, 1, key=i))
        
        with st.container():
            
            with st.form(key="Youssef"):
                submit_button = st.form_submit_button(label='Submit')

                if submit_button:
                    with st.spinner("Training models..."):
                        
                        NN_inputs = classifier_inputs(tt_proportion,
                                                selected_function,
                                                tuple(a),
                                                selected_solver,
                                                iteration_num,
                                                norm_bool
                                                )

                        Classifier = NN_Classifier(cl_df, NN_inputs, col_idx)

                        Classifier.Classify()
                        Classifier.printing()
                        Classifier.Conf()


    #Selected 'Classification (Noah)'
    if cl_method == 'Classification (Noah)':

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                upsample_cl = st.checkbox('Upsample data?')
                scale = st.checkbox('Scale data?')
                del_na = st.checkbox('Get rid of N/A values?')

            with cc2:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)

            with cc3:
                classifier_list = ["KNN", "SVM", "LR"]
                selected_classifier = st.selectbox("Select a classifier:", classifier_list)

                if selected_classifier == "KNN": 
                    k_default = isqrt(cl_df.shape[0])
                    if k_default >= 200: k_default = 200
                    k_value = st.number_input('"k" value:', 1, 200, k_default, 1)

                if selected_classifier == "SVM":
                    kernel_list = ['linear', 'poly', 'rbf', 'sigmoid']
                    selected_kernel = st.selectbox("Kernel:", kernel_list)

                if selected_classifier == "LR":
                    solver_list = ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
                    selected_solver = st.selectbox("Solver:", solver_list)

        with st.container():
            st.subheader("Model")
            class_obj=Classification(cl_df)
            st.dataframe(class_obj.describe_dataframe())

            with st.form(key="Noah"):
                submit_button = st.form_submit_button(label='Submit')

                if submit_button:
                    with st.spinner("Training models..."):
                            
                        try: 
                            if selected_classifier == "KNN":
                                class_obj.split_train_test(y_column_name = selected_column, 
                                                        test_size = tt_proportion,
                                                        random_state = 0, 
                                                        upsample = upsample_cl, 
                                                        scaling = scale, 
                                                        deleting_na = del_na)
                                class_obj.build_classifier('KNN', k_value)
                                class_obj.show_classifier_accuracy()

                            if selected_classifier == "SVM":
                                class_obj.split_train_test(y_column_name = selected_column, 
                                                        test_size = tt_proportion,
                                                        random_state = 0, 
                                                        upsample = upsample_cl, 
                                                        scaling = scale, 
                                                        deleting_na = del_na)
                                class_obj.build_classifier('SVM', selected_kernel)
                                class_obj.show_classifier_accuracy()

                            if selected_classifier == "LR":
                                class_obj.split_train_test(y_column_name = selected_column, 
                                                        test_size = tt_proportion,
                                                        random_state = 0, 
                                                        upsample = upsample_cl, 
                                                        scaling = scale, 
                                                        deleting_na = del_na)
                                class_obj.build_classifier('LR', selected_solver)
                                class_obj.show_classifier_accuracy()
                        except ValueError as e:
                            st.error("Please check if you selected a dataset and column suitable for binary classification. \nAlternatively, your labels should be one-hot encoded.")

    if cl_method == 'Random Forest (Sneha)':

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)
                estimator_value = st.slider('Estimator:', 0, 1000, 500, 10)
                maxim_depth = st.slider('Maximal depth:', 0, 15, 5, 1)


            with cc2:
                columns_list = list(cl_df.columns)
                selected_column = st.selectbox("Column to classify:", columns_list)
                criterion_list = ["gini", "entropy"]
                selected_criterion = st.selectbox("Select a criterion:", criterion_list)

            with cc3:
                depth = st.slider('Max depth:', 1, 10, 5, 1)

                minimum_leaf = st.slider('Min samples leaf:', 0, 15, 3, 1)

                min_split = st.slider('Min samples split:', 0, 15, 2, 1)

            with st.container():
                st.subheader("Model")
                scaler = MinMaxScaler()
                features = [s for s in cl_df.columns if s != selected_column]
                scaled_Dataframe = pd.DataFrame(data = cl_df)
                scaled_Dataframe[features] = scaler.fit_transform(cl_df[features])

                X = scaled_Dataframe[features]  # contains the features
                Y = scaled_Dataframe[selected_column]  # contains the target column 

                obj = Sample(X, Y)

                with st.form(key="Noah"):
                    submit_button = st.form_submit_button(label='Submit')

                    if submit_button:
                        try:    
                            with st.spinner("Training models..."):
                                obj.model(estimator_value, tt_proportion, selected_criterion, depth, minimum_leaf, min_split)
                                obj.report(obj.Y_test, obj.Y_pred)
                                obj.accuracy(obj.Y_test, obj.Y_pred)
                        except ValueError as e:
                            st.error('Something went wrong, Sneha...')
        

if __name__ == "__main__":
    main()
