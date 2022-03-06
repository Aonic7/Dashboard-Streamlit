import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
# from pathlib import Path

from sklearn.model_selection import train_test_split
from .MLP_Classifier import NN_Classifier, classifier_inputs
from .ClassificationClass_Noah import Classification

def import_dset(data_obj):
    try:
        st.write('Try execution')
        cl_df1 = pd.read_csv('Smoothing_and_Filtering//Preprocessing Dataset.csv', index_col=None)

    except:
        st.write('Exception execution')
        cl_df1 = data_obj.df
        st.error("""You did not smooth of filter the data.
                    Please go to 'Smoothing and filtering' and finalize your results.
                    Otherwise, the default dataset would be used!
                    """)    
    return cl_df1


def main(data_obj):

    st.header("Classification")

    # cl_df = import_dset(data_obj)
    # try:
    

    try:
        var_read = pd.read_csv("Smoothing_and_Filtering//Preprocessing dataset.csv")
        cl_df = var_read
        st.dataframe(cl_df)
        # st.write('Try execution')
    except:
        cl_df = data_obj.df
        # st.write('Where is money, Lebovskiy?')

    # except:
    #     st.write('Exception execution')
    #     cl_df = data_obj.df
    #     st.error("""You did not smooth of filter the data.
    #                 Please go to 'Smoothing and filtering' and finalize your results.
    #                 Otherwise, the default dataset would be used!
    #                 """)    
    

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)

    # Main data classification method radio selector
    cl_method = st.radio(label='Classification Method', options=[
                         'Neural Networks', 'Classification (Noah)'])

    # Selected 'Neural Networks'
    if cl_method == 'Neural Networks':

        st.dataframe(cl_df)
        st.write(cl_df.shape)

        with st.container():
            st.subheader('Select input settings')

            cc1, cc2, cc3 = st.columns(3)
            
            with cc1:
                tt_proportion = st.slider('Portion of test data', 0.0, 1.0, 0.2, 0.05)

                iteration_num = st.slider('Number of iterations', 100, 500, 200, 50)

                norm_bool = st.checkbox('Normalize data?')              
                
            
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
                    a.append(st.number_input(f'Number of neurons in hidden layer {i+1}:', 1, 20, 1, 1, key=i))


        # with st.container():
        #     # Splitting the data into training and test sets
        #     train_df, test_df = train_test_split(cl_df, test_size=tt_proportion, random_state=109)

        #     # Using numpy to create arrays of lables and features
        #     train_labels = np.array(train_df.pop(selected_column))
        #     test_labels = np.array(test_df.pop(selected_column))
        #     train_features = np.array(train_df)
        #     test_features = np.array(test_df)

        #     # Scaling the features using Standard Scaler
        #     if norm_bool == True:
        #         scaler = MinMaxScaler()
        #         train_features = scaler.fit_transform(train_features)
        #         test_features = scaler.transform(test_features)

        #     # Having a look at the results
        #     st.write('Training labels shape:', train_labels.shape)
        #     st.write('Test labels shape:', test_labels.shape)
        #     st.write('Training features shape:', train_features.shape)
        #     st.write('Test features shape:', test_features.shape)

        #     sme = SMOTEENN(random_state=109, sampling_strategy=0.48)
        #     X_res, y_res = sme.fit_resample(train_features, train_labels)

        #     clf_NN = MLPClassifier(hidden_layer_sizes = tuple(a), 
        #                        activation = selected_function,
        #                        solver = selected_solver, 
        #                        learning_rate = 'adaptive',
        #                        max_iter = 500,
        #                        random_state = 109,
        #                        shuffle=True,
        #                        batch_size=15,
        #                        alpha=0.0005
        #                        )

        #     clf_NN = clf_NN.fit(X_res, y_res)

        #     y_pred_NN = clf_NN.predict(test_features)
        #     st.markdown(classification_report(test_labels,y_pred_NN), unsafe_allow_html=True)


        with st.container():
            NN_inputs = classifier_inputs(tt_proportion,
                                       selected_function,
                                       tuple(a),
                                       selected_solver,
                                       iteration_num,
                                       norm_bool
                                       )

            Classifier = NN_Classifier(cl_df, NN_inputs, col_idx)

            Classifier.Classify()

            #st.write(getattr(Classifier.classifier_outputs, 'Report'))
            Classifier.printing()
            Classifier.Conf()

    if cl_method == 'Classification (Noah)':
        
        st.dataframe(cl_df)
        st.write(cl_df.shape)

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
                    k_value = st.number_input('"k" value:', 1, 10, 5, 1)

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

                    
        

if __name__ == "__main__":
    main()
