import pandas as pd
import streamlit as st
from ClassificationClass import Classification
from sklearn.metrics import classification_report

from math import isqrt


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

    st.write(
        '<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>',
        unsafe_allow_html=True)

    # Main data classification method radio selector

    cl_df = data_obj.df
    st.dataframe(cl_df)
    class_obj = Classification(cl_df)

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
        cc1, cc2, cc3 = st.columns(3)

        with cc1:
            st.subheader('Model Training')
            submit_button = st.button(label='Train Model')

            if submit_button:
                with st.spinner("Training models..."):

                    try:
                        if selected_classifier == "KNN":
                            class_obj.split_train_test(y_column_name=selected_column,
                                                       test_size=tt_proportion,
                                                       random_state=0,
                                                       upsample=upsample_cl,
                                                       scaling=scale,
                                                       deleting_na=del_na)
                            st.session_state.model, \
                            st.session_state.pred =class_obj.build_classifier('KNN', k_value)

                            st.session_state.fig_class,\
                            st.session_state.cnf_matrix,\
                            st.session_state.report = class_obj.show_classifier_accuracy()

                        if selected_classifier == "SVM":
                            class_obj.split_train_test(y_column_name=selected_column,
                                                       test_size=tt_proportion,
                                                       random_state=0,
                                                       upsample=upsample_cl,
                                                       scaling=scale,
                                                       deleting_na=del_na)
                            st.session_state.model, \
                            st.session_state.pred =class_obj.build_classifier('SVM', selected_kernel)

                            st.session_state.fig_class,\
                            st.session_state.cnf_matrix,\
                            st.session_state.report = class_obj.show_classifier_accuracy()

                        if selected_classifier == "LR":
                            class_obj.split_train_test(y_column_name=selected_column,
                                                       test_size=tt_proportion,
                                                       random_state=0,
                                                       upsample=upsample_cl,
                                                       scaling=scale,
                                                       deleting_na=del_na)
                            st.session_state.model, \
                            st.session_state.pred =class_obj.build_classifier('LR', selected_solver)

                            st.session_state.fig_class,\
                            st.session_state.cnf_matrix,\
                            st.session_state.report = class_obj.show_classifier_accuracy()

                    except ValueError as e:
                        st.error(
                            "Please check if you selected a dataset and column suitable for binary classification. \nAlternatively, your labels should be one-hot encoded.")
            try:
                st.metric("Overall Accuracy of "+str(st.session_state.model),st.session_state.pred)
                for dim in range(st.session_state.cnf_matrix.shape[0]):
                    accuracy = st.session_state.cnf_matrix[dim][dim] / sum(st.session_state.cnf_matrix[dim])
                    # st.write("Accuracy for the ",dim,"class:", accuracy)
                    # st.write(f"Accuracy for the {dim} class:")
                    st.metric(f"Accuracy for the {dim} class:", round(accuracy, 4))
            except:
                pass

        with cc2:
            st.subheader('Model Graphs')
            try:
                st.caption("Confusion Matrix")
                st.pyplot(st.session_state.fig_class)
                st.caption("Classification Report")
                st.dataframe(st.session_state.report)
            except:
                pass

        with cc3:
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

if __name__ == "__main__":
    main()