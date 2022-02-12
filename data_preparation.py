import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np

def linePlot_Out_recogn(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe.df[column]))], data = dataframe.df)
    st.pyplot(fig)

def BoxPlot(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(dataframe[column])
    st.pyplot(fig)

def data_preparation_run(data_obj):
    st.header("DATA PREPARATION")
    st.subheader('Remove outlier')
    st.dataframe(data_obj.df)

    with st.form("Data Preparation parameters selector"):
        columns_list = list(data_obj.df.select_dtypes(exclude=['object']).columns)
        std_coeff = st.number_input("Enter standard deviation coefficient (multiplier) ", 0.0, 3.1, 2.0, 0.1)
        selected_column = st.selectbox("Select a column", columns_list)

        submitted = st.form_submit_button("Create a plot")
        if submitted:
            st.write("Standard deviation", std_coeff, "Column", selected_column)
    
    
    rm_outlier = removeOutlier(data_obj, selected_column, std_coeff)
    BoxPlot(rm_outlier, selected_column)

    rm_outlier.to_csv("Prepared Dataset.csv")




def removeOutlier (data_obj, columnName, n):
    mean = data_obj.df[columnName].mean()
    std = data_obj.df[columnName].std()  
    fromVal = mean - n * std 
    toVal = mean + n * std 
    filtered = data_obj.df[(data_obj.df[columnName] >= fromVal) & (data_obj.df[columnName] <= toVal)] #apply the filtering formula to the column
    return filtered 

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

