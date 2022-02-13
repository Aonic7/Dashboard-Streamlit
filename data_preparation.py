import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import seaborn as sns
import numpy as np
import scipy.stats as stats

def linePlot_Out_recogn(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.lineplot(y = column, x = [i for i in range(len(dataframe.df[column]))], data = dataframe.df)
    st.pyplot(fig)

def BoxPlot(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.boxplot(y = dataframe[column], orient='v')
    st.pyplot(fig)

def DoubleBoxPlot(initdf, dataframe, column):
    fig, axes = plt.subplots(2,1, sharex=True, figsize=(10,4))
    sns.boxplot(initdf[column], ax=axes[0], color='skyblue', orient="v")
    axes[0].set_title("Original dataframe")
    sns.boxplot(dataframe[column], ax=axes[1], color='green', orient="v")
    axes[1].set_title("Resulting dataframe")
    fig.tight_layout()
    st.pyplot(fig)

def Histogram(dataframe, column):
    fig = plt.figure(figsize=(10, 4))
    sns.histplot(data=dataframe, x=column)
    st.pyplot(fig)

def removeOutlier(df, columnName, n):
    mean = df[columnName].mean()
    std = df[columnName].std()  
    fromVal = mean - n * std 
    toVal = mean + n * std 
    filtered = df[(df[columnName] >= fromVal) & (df[columnName] <= toVal)] #apply the filtering formula to the column
    return filtered

def removeOutlier_q(df, columnName, n1, n2):
    lower_quantile, upper_quantile = df[columnName].quantile([n1, n2]) #quantiles are generally expressed as a fraction (from 0 to 1)
    filtered = df[(df[columnName] > lower_quantile) & (df[columnName] < upper_quantile)]
    return filtered

def removeOutlier_z(df, columnName, n):
    z = np.abs(stats.zscore(df[columnName])) #find the Z-score for the column
    filtered = df[(z < n)] #apply the filtering formula to the column
    return filtered #return the filtered dataset



def data_preparation_run(data_obj):
    st.header("DATA PREPARATION")

    if st.sidebar.button("Reset dataframe to the initial one"):
        data_obj.df.to_csv("Prepared Dataset.csv", index=False)

    if pd.read_csv('Prepared Dataset.csv').shape[0] < data_obj.df.shape[0]:
        current_df = pd.read_csv('Prepared Dataset.csv', index_col = None)
    else:
        current_df = data_obj.df

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader('Original dataframe')
        st.dataframe(data_obj.df)
        st.write(data_obj.df.shape)

    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)
    
    dp_method = st.radio(label = 'Data Prep Method', options = ['Remove outliers','Interpolation','Smoothing'])
    
    if dp_method == 'Remove outliers':
        rmo_radio = st.radio(label = 'Remove outliers method',
                             options = ['Std','Q','Z'])

        if rmo_radio == 'Std':


            with st.container():
                st.subheader('Remove outliers using standard deviation')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    rm_outlier = removeOutlier(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-rm_outlier.shape[0]} rows will be removed.')
                        
                if bp:
                    #BoxPlot(rm_outlier.reset_index(drop=True), selected_column)
                    DoubleBoxPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(rm_outlier.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = rm_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

        if rmo_radio == 'Q':

            with st.container():
                st.subheader('Remove outliers using quantiles')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    q_values = st.slider('Select a range of quantiles',
                                        0.00, 1.00, (0.25, 0.75))
                    selected_column = st.selectbox("Select a column:", columns_list)
                    q_outlier = removeOutlier_q(current_df, selected_column, q_values[0], q_values[1])
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-q_outlier.shape[0]} rows will be removed.')
                        
                if bp:
                    DoubleBoxPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(q_outlier.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = q_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

        if rmo_radio == 'Z':

            with st.container():
                st.subheader('Remove outliers using Z-score')

                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient: ", 0.0, 3.1, 2.0, 0.1)
                    #ask Marina about min and max values
                    selected_column = st.selectbox("Select a column:", columns_list)
                    z_outlier = removeOutlier_z(current_df, selected_column, std_coeff)
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-z_outlier.shape[0]} rows will be removed.')
                        
                if bp:
                    DoubleBoxPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column)

                if hist:
                    Histogram(z_outlier.reset_index(drop=True), selected_column)

                #current_df = rm_outlier.reset_index(drop=True)
                if st.button("Save remove outlier results"):
                    current_df = z_outlier.reset_index(drop=True)
                    current_df.to_csv("Prepared Dataset.csv", index=False)

    if dp_method == 'Smoothing':
        rmo_radio = st.radio(label = 'Smoothing',
                             options = ['Median filter','Moving average','Savitzky Golay'])

    with col2:
        st.subheader('Resulting dataframe')
        if dp_method == 'Remove outliers' and rmo_radio == 'Std':
            st.dataframe(rm_outlier.reset_index(drop=True))
            st.write(rm_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Q':
            st.dataframe(q_outlier.reset_index(drop=True))
            st.write(q_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Z':
            st.dataframe(z_outlier.reset_index(drop=True))
            st.write(z_outlier.shape)
    
    with col3:
        st.subheader('Current dataframe')
        st.dataframe(current_df)
        st.write(current_df.shape)
    










































# def data_preparation_run(data_obj):
#     st.header("DATA PREPARATION")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader('Original dataframe')
#         st.dataframe(data_obj.df)
#         st.write(data_obj.df.shape)

#     if 'current_df' not in st.session_state:
#         if data_obj.df.shape[0] == pd.read_csv("Prepared Dataset.csv").shape[0]:
#             st.session_state.current_df = data_obj.df
#         else:
#             st.session_state.current_df = pd.read_csv("Prepared Dataset.csv")
#         st.write('Blah')

#     if st.sidebar.button("Reset dataframe to the initial one"):
#         del st.session_state['current_df']
#         st.session_state.current_df = data_obj.df



#     st.subheader('Select Data Preparation Methods:')
#     with st.expander("Remove outliers"):
#         st.subheader('Remove outliers')

#         with st.form("Data Preparation parameters selector"):
#             columns_list = list(st.session_state.current_df.select_dtypes(exclude=['object']).columns)
#             std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
#             selected_column = st.selectbox("Select a column:", columns_list)
#             graph = st.radio('Select a graph:', ['Boxplot', 'Histogram', 'Update', 'Reset'])
#             submitted = st.form_submit_button("Create a plot")

#             if submitted:
#                 st.write("Standard deviation: ", round(std_coeff, 2), "   Column: ", selected_column)

#                 if graph == 'Boxplot':
#                     rm_outlier = removeOutlier(st.session_state.current_df, selected_column, std_coeff)
#                     BoxPlot(rm_outlier, selected_column)

#                     st.write(f'{st.session_state.current_df.shape[0]-rm_outlier.shape[0]} rows will be dropped.')
#                     st.dataframe(rm_outlier)
#                     st.write(rm_outlier.shape)
#                     st.write(st.session_state.current_df.shape)
#                     rm_outlier.to_csv("Prepared Dataset.csv", index=False)

#                 elif graph == 'Histogram':
#                     rm_outlier = removeOutlier(st.session_state.current_df, selected_column, std_coeff)
#                     Histogram(rm_outlier, selected_column)

#                     st.write(f'{st.session_state.current_df.shape[0]-rm_outlier.shape[0]} rows will be dropped.')
#                     st.dataframe(rm_outlier)
#                     st.write(rm_outlier.shape)
#                     st.write(st.session_state.current_df.shape)
#                     rm_outlier.to_csv("Prepared Dataset.csv", index=False)

#                 elif graph == 'Update':
#                     st.write("Updated!")

#                 elif graph == 'Reset':
#                     del st.session_state['current_df']

#                 #st.write(st.session_state.current_df.shape)
#                 #st.session_state.current_df.to_csv("Prepared Dataset.csv", index=False)


#     with col2:
#         st.subheader('Current dataframe')
#         st.dataframe(st.session_state.current_df)
#         st.write(st.session_state.current_df.shape)

























































# def data_preparation_run(data_obj):
#     st.header("DATA PREPARATION")

#     col1, col2 = st.columns(2)
#     with col1:
#         st.subheader('Original dataframe')
#         st.dataframe(data_obj.df)
#         st.write(data_obj.df.shape)

#     if 'current_df' not in st.session_state:
#         st.session_state.current_df = data_obj.df
#         st.session_state.current_df.to_csv("Prepared Dataset.csv", index=False)
#         st.write('Blah')
#     else:
#         st.session_state.current_df = pd.read_csv("Prepared Dataset.csv")

#     if st.sidebar.button("Reset dataframe to the initial one"):
#         del st.session_state['current_df']
#         st.session_state.current_df = data_obj.df

#     with col2:
#         st.subheader('Current dataframe')
#         st.dataframe(st.session_state.current_df)
#         st.write(st.session_state.current_df.shape)

#     st.subheader('Select Data Preparation Methods:')
#     with st.expander("Remove outliers"):
#         st.subheader('Remove outliers')

#         with st.form("Data Preparation parameters selector"):
#             columns_list = list(st.session_state.current_df.select_dtypes(exclude=['object']).columns)
#             std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
#             selected_column = st.selectbox("Select a column:", columns_list)
#             graph = st.radio('Select a graph:', ['Boxplot', 'Histogram', 'Update', 'Reset'])
#             #rm_outlier = removeOutlier(st.session_state.current_df, selected_column, std_coeff)
#             submitted = st.form_submit_button("Create a plot")

#             if submitted:
#                 st.write("Standard deviation: ", round(std_coeff, 2), "   Column: ", selected_column)
#                 rm_outlier = removeOutlier(st.session_state.current_df, selected_column, std_coeff)

#                 if graph == 'Boxplot':
#                     BoxPlot(rm_outlier, selected_column)

#                     st.write(f'{st.session_state.current_df.shape[0]-rm_outlier.shape[0]} rows were dropped.')
#                     st.dataframe(rm_outlier)
#                     st.write(rm_outlier.shape)

#                     rm_outlier.to_csv("Prepared Dataset.csv", index=False)
#                     st.write(st.session_state.current_df.shape)

#                 elif graph == 'Histogram':
#                     Histogram(rm_outlier, selected_column)

#                     st.write(f'{st.session_state.current_df.shape[0]-rm_outlier.shape[0]} rows were dropped.')
#                     st.dataframe(rm_outlier)
#                     st.write(rm_outlier.shape)

#                     rm_outlier.to_csv("Prepared Dataset.csv", index=False)
#                     st.write(st.session_state.current_df.shape)

#                 elif graph == 'Update':
#                     st.write("Updated!")

#                 elif graph == 'Reset':
#                     del st.session_state['current_df']





















    # if st.button('Apply data preparation changes'):
    #     st.session_state.current_df.to_csv("Prepared Dataset.csv", index=False)
    #     del st.session_state['current_df']
    #     st.write('Saved!')
    # else:
    #     st.write('Your work has not been saved!')


























    # st.subheader('Select Data Preparation Methods:')
    # with st.expander("Remove outliers"):
    #     st.subheader('Remove outliers')

    #     with st.form("Data Preparation parameters selector"):
    #         columns_list = list(data_obj.df.select_dtypes(exclude=['object']).columns)
    #         std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
    #         selected_column = st.selectbox("Select a column:", columns_list)
    #         graph = st.radio('Select a graph:', ['Boxplot', 'Histogram'])
    #         rm_outlier = removeOutlier(data_obj, selected_column, std_coeff)
    #         submitted = st.form_submit_button("Create a plot")

    #         if submitted:
    #             st.write("Standard deviation: ", round(std_coeff, 2), "   Column: ", selected_column)
    #             if graph == 'Boxplot':
    #                 BoxPlot(rm_outlier, selected_column)
    #             elif graph == 'Histogram':
    #                 Histogram(rm_outlier, selected_column)

    #             st.write(f'{data_obj.df.shape[0]-rm_outlier.shape[0]} rows were dropped.')
    #             st.dataframe(rm_outlier)
    #             st.write(rm_outlier.shape)

    #             rm_outlier.to_csv("Prepared Dataset.csv", index=False)





# syncMachine = pd.read_csv("D:\MAIT21\OOP\Data\Regression\synchronous_machine.csv", delimiter=';', decimal=',')
# syncMachine = removeOutlier(syncMachine, 'If', 2)
# print(f'Dataframe size after filtering = {syncMachine.size}') # also we can print the size of the dataset after filtering
# syncMachine.If.hist() # we can plot the histogram after filtering
# plt.show()
