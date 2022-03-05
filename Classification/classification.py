from ast import Not
from code import interact
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
from io import StringIO
from pandas.api.types import is_numeric_dtype
from .MLP_Classifier import NN_Classifier, classifier_inputs
from typing import NamedTuple

import numpy as np

from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from sklearn.utils.multiclass import unique_labels
from tabulate import tabulate

def classification_report(y_true, y_pred, labels=None, target_names=None,
                          sample_weight=None, digits=4, tablfmt='pipe'):
    """  Better format for sklearn's classification report
    Based on tabulate package
    ----------
    y_true : 1d array-like, or label indicator array / sparse matrix
        Ground truth (correct) target values.
    y_pred : 1d array-like, or label indicator array / sparse matrix
        Estimated targets as returned by a classifier.
    labels : array, shape = [n_labels]
        Optional list of label indices to include in the report.
    target_names : list of strings
        Optional display names matching the labels (same order).
    sample_weight : array-like of shape = [n_samples], optional
        Sample weights.
    digits : int
        Number of digits for formatting output floating point values
    Returns
    -------
    report : string
        Text summary of the precision, recall, F1 score for each class.
        The reported averages are a prevalence-weighted macro-average across
        classes (equivalent to :func:`precision_recall_fscore_support` with
        ``average='weighted'``).
        Note that in binary classification, recall of the positive class
        is also known as "sensitivity"; recall of the negative class is
        "specificity".
    Examples
    --------
    >>> from sklearn.metrics import classification_report
    >>> y_true = [0, 1, 2, 2, 2]
    >>> y_pred = [0, 0, 2, 2, 1]
    >>> target_names = ['class 0', 'class 1', 'class 2']
    >>> print(classification_report(y_true, y_pred, target_names=target_names))
                 precision    recall  f1-score   support
    """
    floatfmt = '.{:}f'.format(digits)
    if labels is None:
        labels = unique_labels(y_true, y_pred)
    else:
        labels = np.asarray(labels)

    if target_names is not None and len(labels) != len(target_names):
        print(
            "labels size, {0}, does not match size of target_names, {1}"
            .format(len(labels), len(target_names))
        )

    last_line_heading = 'avg / total'

    if target_names is None:
        target_names = [u'%s' % l for l in labels]

    headers = ["precision", "recall", "f1-score", "support"]

    p, r, f1, s = precision_recall_fscore_support(y_true, y_pred,
                                                  labels=labels,
                                                  average=None,
                                                  sample_weight=sample_weight)

    rows = zip(target_names, p, r, f1, s)
    tbl_rows = []
    for row in rows:
        tbl_rows.append(row)

    # compute averages
    last_row = (last_line_heading,
                np.average(p, weights=s),
                np.average(r, weights=s),
                np.average(f1, weights=s),
                np.sum(s))
    tbl_rows.append(last_row)
    return tabulate(tbl_rows, headers=headers,
                    tablefmt=tablfmt, floatfmt=floatfmt)

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
            st.write(type(tt_proportion))

            Classifier = NN_Classifier(cl_df, NN_inputs, col_idx)

            Classifier.Classify()
            # st.write(Classifier.Classify()[52])
            Classifier.printing()
            

            Classifier.Conf()
            # #st.write(classification_report(getattr(Classifier, 'NN_Outputs.NN_Inputs')))

            # class boris(NamedTuple):
            #     test_size:  str

            # inputs1 = boris("Sasay kudasai")

            # st.write(inputs1[0])
            # st.write(Classifier.NN_Outputs['y_pred'])
            # st.write("Blah")
            # st.write(Classifier.NN_Outputs.y_pred)
            # st.write(getattr(Classifier.classifier_outputs, 'X_test'))
            # st.write(getattr(Classifier.classifier_outputs, 'Error_message'))
            # st.write(getattr(Classifier.classifier_outputs, 'Report'))
            # print(getattr(Classifier.classifier_outputs, 'Report'))

            # from operator import itemgetter as _itemgetter
            # f = _itemgetter(1)
            # st.write(Classifier.NN_Outputs[0])
                 

if __name__ == "__main__":
    main()
