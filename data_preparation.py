import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np

def linePlot_Out_recogn(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe.df[column]))], data = dataframe.df)
    st.pyplot(fig)

def data_preparation_run(data_obj):
    st.header("DATA PREPARATION")
    st.write('Remove outlier')
    st.dataframe(data_obj.df)

    columns_list = list(data_obj.df.select_dtypes(exclude=['object']).columns)
    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier) ", 0.0, 3.0, 2.0, 0.1)
    selected_column = st.selectbox("Select a column", columns_list)
    
    
    linePlot_Out_recogn(data_obj, selected_column)


def removeOutlier (df, columnName, n):
    mean = df[columnName].mean() #find the mean for the column
    std = df[columnName].std()  #find standard deviation for column
    print(f'Mean = {mean}, std = {std}') #we can print these 2 values
    print(f'Dataframe size = {df.size}') # also we can print initial size of the dataset
    fromVal = mean - n * std  # find the min value of the filtering boundary (by default n=2)
    toVal = mean + n * std # find the max value of the filtering boundary (by default n=2)
    print(f'Valid values from {fromVal} to {toVal}') # we can print the filtering boundaries
    filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
    return filtered #return the filtered dataset

# syncMachine = pd.read_csv("D:\MAIT21\OOP\Data\Regression\synchronous_machine.csv", delimiter=';', decimal=',')
# syncMachine = removeOutlier(syncMachine, 'If', 2)
# print(f'Dataframe size after filtering = {syncMachine.size}') # also we can print the size of the dataset after filtering
# syncMachine.If.hist() # we can plot the histogram after filtering
# plt.show()


def data_prep_outlier_inputs():
    global column_select_or, std_coeff
    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier) ", 0.0, 3.0, 2.0, 0.1)
    columns_list = list(df.select_dtypes(exclude=['object']).columns)   
    with st.form(key='outlier form'):
        column_select_or = st.selectbox(
            label='Select a column',
            options=columns_list
        )
        submit_button = st.form_submit_button(label='Submit')
    if submit_button:
        return column_select_or, std_coeff

