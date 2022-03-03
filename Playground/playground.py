import pandas as pd #for working with dataframes
import numpy as np #for numerical and vector operations
import streamlit as st #streamlit backend

# Importing machine learning tools for preprocessing
from sklearn.model_selection import train_test_split #for splitting the data into training and test sets
from sklearn.preprocessing import StandardScaler #for feature scaling
from sklearn.preprocessing import MinMaxScaler

#from pycaret.classification import *
from imblearn.combine import SMOTEENN
from sklearn.neural_network import MLPClassifier

def main(data_obj):
    st.header("Scaling and Train-test split")
    st.subheader("Select the dataset to work with:")

    # Horizontal styling of radio buttons
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)
    
    # Dataset selector
    dp_method = st.radio(label = 'Dataset', options = ['Smoothed and filtered','Original'])

    if dp_method == 'Smoothed and filtered':
        try:
            current_df = pd.read_csv('Smoothing_and_Filtering//Preprocessing dataset.csv', index_col = None)
        except:
            st.error("""You did not smooth of filter the data.
                        Please go to 'Smoothing and filtering' and finalize your results.
                        Otherwise, the default dataset would be used!
                     """)
            current_df = data_obj.df
    
    if dp_method == 'Original':
        current_df = data_obj.df

    st.write('Dataframe that will be used for further processing')
    st.dataframe(current_df)
    st.write(current_df.shape)
    










    

    label_column = 'Select column that contains labels (no scaling will be applied to it)'
    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
    selected_column = st.selectbox(label_column, columns_list)

    # st.write(current_df.drop(selected_column, axis=1))

    scaler_method = st.radio(label = 'Scaler', options = ['Standard scaler', 'MinMax scaler', 'No scaler'])

    if scaler_method == 'Standard scaler':
        scaler_func = StandardScaler()
        scaled_df = scaler_func.fit_transform(current_df.drop(selected_column, axis=1))
        st.write('Standard scaler was applied')
        st.write(type(scaled_df))
        st.dataframe(scaled_df)

    if scaler_method == 'MinMax scaler':
        scaler_func = MinMaxScaler()
        scaled_df = scaler_func.fit_transform(current_df.drop(selected_column, axis=1))
        st.write('MinMax scaler was applied')
        st.dataframe(scaled_df)

    if scaler_method == 'No scaler':
        st.write('No scaler method will be applied, proceed to the next step.')


    with st.container():
        st.subheader('PyCaret Classification')
        # exp_clf01 = setup(data = current_df,
        #                   target = selected_column,
        #                   session_id = 123,
        #                   silent=True,
        #                   fix_imbalance=True)
        # with st.spinner("Training models..."):
        #     best = compare_models(n_select = 5, sort="MCC")
        #     st.write(pull())

    with st.container():
        st.subheader('Neural Networks')
        # Splitting the data into training and test sets
        train_df, test_df = train_test_split(current_df, test_size=0.2, random_state=0)

        # Using numpy to create arrays of lables and features
        train_labels = np.array(train_df.pop(selected_column))
        test_labels = np.array(test_df.pop(selected_column))
        train_features = np.array(train_df)
        test_features = np.array(test_df)

        # Scaling the features using Standard Scaler
        scaler = StandardScaler()
        train_features = scaler.fit_transform(train_features)
        test_features = scaler.transform(test_features)

        # Having a look at the results
        st.write('Training labels shape:', train_labels.shape)
        st.write('Test labels shape:', test_labels.shape)
        st.write('Training features shape:', train_features.shape)
        st.write('Test features shape:', test_features.shape)


        sme = SMOTEENN(random_state=42, sampling_strategy=0.48)
        X_res, y_res = sme.fit_resample(train_features, train_labels)

        clf_NN = MLPClassifier(hidden_layer_sizes = (150,100,50), 
                               activation = 'logistic',
                               solver = 'adam', 
                               learning_rate = 'adaptive',
                               max_iter = 500,
                               random_state = 42,
                               shuffle=True,
                               batch_size=15,
                               alpha=0.0005
                               )





if __name__ == "__main__":
   main()


    
    





# Splitting the data into training and test sets
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=0)

# Using numpy to create arrays of lables and features
#train_labels = np.array(train_df.pop('Production'))
#test_labels = np.array(test_df.pop('Production'))
# train_features = np.array(train_df)
# test_features = np.array(test_df)

# # Scaling the features using Standard Scaler
# scaler = StandardScaler()
# train_features = scaler.fit_transform(train_features)
# test_features = scaler.transform(test_features)

# # Having a look at the results
# print('Training labels shape:', train_labels.shape)
# print('Test labels shape:', test_labels.shape)
# print('Training features shape:', train_features.shape)
# print('Test features shape:', test_features.shape)