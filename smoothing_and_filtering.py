# General import section
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import numpy as np #to work with arrays and series
import os
from io import StringIO

from scipy.signal import medfilt

# Visualization import section
from visualization import doubleLinePlot, DoubleBoxPlot, Histogram, ScatterPlot

# Remove outliers import
from dp_remove_outliers import removeOutlier, removeOutlier_q, removeOutlier_z

def import_dset(data_obj):
    try:
        a = pd.read_csv('Filtered Dataset.csv', index_col = None)
        if a.equals(data_obj.df) == False:
            current_df = a
            #st.sidebar.write("1")
            #st.sidebar.write(a.equals(data_obj.df))
        else:
            current_df = data_obj.df
            #st.sidebar.write("2")
    except:
        current_df = data_obj.df
        #st.sidebar.write("3")

    return current_df

def moving_average(dataframe, column, filter_length):
    df_prep = dataframe.copy()
    tair_moving_average = np.convolve(dataframe[column], np.ones((filter_length)), mode ="same")
    tair_moving_average /= filter_length
    tair_moving_average= tair_moving_average[round(filter_length/2): round(len(dataframe[column])-filter_length/2)]
    df_prep[column] = tair_moving_average


    return df_prep

def median_filter(dataframe, column, filter_length):
    """
    data: function that will be filtered
    filter_length: length of the window
    """
    s = dataframe.copy()
    # medfilt_tair = medfilt(dataframe[column], filter_lenght)
    # filtered = dataframe[(dataframe[column] == medfilt_tair)] 
    # # s = pd.DataFrame(medfilt_tair)
    # # s.columns=[column]
    # return filtered

    medfilt_tair = medfilt(dataframe[column], filter_length)
    s[column] = medfilt_tair
    #s = pd.DataFrame(medfilt_tair)
    #s.columns=[column]
    return s



def smoothing_and_filtering_run(data_obj):
    st.header("Smoothing and filtering")

    # A button to circumvent loading the dataset using in the last session
    if st.sidebar.button("Reset dataframe to the initial one"):
        data_obj.df.to_csv("Filtered Dataset.csv", index=False)

    # 1
    # Loading the appropriate dataset
    #if pd.read_csv('Filtered Dataset.csv').shape[0] < data_obj.df.shape[0]:
        #current_df = pd.read_csv('Filtered Dataset.csv', index_col = None)
    #else:
        #current_df = data_obj.df

    # 2
    # try:
    #     if pd.read_csv('Filtered Dataset.csv').shape[0] < data_obj.df.shape[0]:
    #         current_df = pd.read_csv('Filtered Dataset.csv', index_col = None)
    #     else:
    #         current_df = data_obj.df
    # except:
    #     current_df = data_obj.df

    current_df = import_dset(data_obj)

    # Overview of the dataframes
    col1, col2, col3 = st.columns(3)

    # Original dataframe display
    with col1:
        st.subheader('Original dataframe')
        st.dataframe(data_obj.df)
        st.write(data_obj.df.shape)

    # Horizontal styling of radio buttons
    st.write('<style>div.row-widget.stRadio > div{flex-direction:row;justify-content: center;}</style>', unsafe_allow_html=True)
    
    # Main data preparation method radio selector
    dp_method = st.radio(label = 'Filtering Method', options = ['Remove outliers','Smoothing','Interpolation'])
    
    # Selected 'Remove outliers'
    if dp_method == 'Remove outliers':

        # Specifying remove outliers submethods
        rmo_radio = st.radio(label = 'Remove outliers method',
                             options = ['Std','Q','Z'])

        # Standard deviation method selected
        if rmo_radio == 'Std':

            # Standard deviation method main
            with st.container():
                st.subheader('Remove outliers using standard deviation')

                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient (multiplier): ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    rm_outlier = removeOutlier(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-rm_outlier.shape[0]} rows will be removed.')
                
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, rm_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(rm_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (std)"):
                    current_df = rm_outlier.reset_index(drop=True)
                    current_df.to_csv("Filtered Dataset.csv", index=False)

        # Quantile range method selected
        if rmo_radio == 'Q':

            # Quantile range method main
            with st.container():
                st.subheader('Remove outliers using quantiles')

                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    q_values = st.slider('Select a range of quantiles',
                                        0.00, 1.00, (0.25, 0.75))
                    selected_column = st.selectbox("Select a column:", columns_list)
                    q_outlier = removeOutlier_q(current_df, selected_column, q_values[0], q_values[1])
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-q_outlier.shape[0]} rows will be removed.')
                        
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, q_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(q_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (Q)"):
                    current_df = q_outlier.reset_index(drop=True)
                    current_df.to_csv("Filtered Dataset.csv", index=False)

        # Z-score method selected
        if rmo_radio == 'Z':

            # Z-score method main
            with st.container():
                st.subheader('Remove outliers using Z-score')

                cc1, cc2, cc3, cc4 = st.columns(4)

                # Input settings
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    std_coeff = st.number_input("Enter standard deviation coefficient: ", 0.0, 3.1, 2.0, 0.1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    z_outlier = removeOutlier_z(current_df, selected_column, std_coeff)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    scatter_plot = st.button('Scatter plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    scatter_column2 = st.selectbox('Select 2nd column for the scatter plot', [s for s in columns_list if s != selected_column])
                with cc4:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-z_outlier.shape[0]} rows will be removed.')
                        
                # Plotting
                if scatter_plot:
                    ScatterPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column, scatter_column2)
                if bp:
                    DoubleBoxPlot(data_obj.df, z_outlier.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(z_outlier.reset_index(drop=True), selected_column)

                # Save results to csv
                if st.button("Save intermediate remove outlier results (Z)"):
                    current_df = z_outlier.reset_index(drop=True)
                    current_df.to_csv("Filtered Dataset.csv", index=False)



    if dp_method == 'Smoothing':
        smooth_radio = st.radio(label = 'Smoothing',
                             options = ['Median filter','Moving average','Savitzky Golay'])
        if smooth_radio == 'Median filter':
            
            with st.container():
                st.subheader('Median filter')

                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 3, 7, 5, 2)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    median_filt = median_filter(current_df, selected_column, filter_len)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                # with cc3:
                #     st.write(" ")
                #     st.write(" ")
                #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
                if plot_basic:
                    doubleLinePlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                    # st.dataframe(median_filt)
                    # st.write(data_obj.df[selected_column].value_counts(ascending=False))
                    # st.write(median_filt[selected_column].value_counts(ascending=False))
                    # st.write("Blah")
                    # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
                    # lf = pd.DataFrame(data=l)
                    # st.write(lf.value_counts(ascending=False))
                if bp:
                    DoubleBoxPlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(median_filt.reset_index(drop=True), selected_column)

                if st.button("Save intermediate smoothing results (median filter)"):
                    current_df = median_filt.reset_index(drop=True)
                    current_df.to_csv("Filtered Dataset.csv", index=False)

        if smooth_radio == 'Moving average':
            with st.container():
                st.subheader('Moving average')

                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 3, 7, 5, 2)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    # median_filt = median_filter(current_df, selected_column, filter_len)
                    moving_ave = moving_average(current_df, selected_column, filter_len)
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-moving_ave.shape[0]} rows will be removed.')
                
                if plot_basic:
                    doubleLinePlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                    # st.dataframe(median_filt)
                    # st.write(data_obj.df[selected_column].value_counts(ascending=False))
                    # st.write(median_filt[selected_column].value_counts(ascending=False))
                    # st.write("Blah")
                    # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
                    # lf = pd.DataFrame(data=l)
                    # st.write(lf.value_counts(ascending=False))
                if bp:
                    DoubleBoxPlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(moving_ave.reset_index(drop=True), selected_column)

                if st.button("Save remove outlier results"):
                    current_df = moving_ave.reset_index(drop=True)
                    current_df.to_csv("Filtered Dataset.csv", index=False)

    # Current dataframe display
    with col2:
        st.subheader('Current dataframe')
        st.dataframe(current_df)
        st.write(current_df.shape)

    # Resulting dataframe display
    with col3:
        st.subheader('Resulting dataframe')

        # For each 'Remove outliers' case
        if dp_method == 'Remove outliers' and rmo_radio == 'Std':
            st.dataframe(rm_outlier.reset_index(drop=True))
            st.write(rm_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Q':
            st.dataframe(q_outlier.reset_index(drop=True))
            st.write(q_outlier.shape)
        if dp_method == 'Remove outliers' and rmo_radio == 'Z':
            st.dataframe(z_outlier.reset_index(drop=True))
            st.write(z_outlier.shape)

        # For each 'Smoothing' case
        if dp_method == 'Smoothing' and smooth_radio == 'Median filter':
            st.dataframe(median_filt.reset_index(drop=True))
            st.write(median_filt.shape)
        if dp_method == 'Smoothing' and smooth_radio == 'Moving average':
            st.dataframe(moving_ave.reset_index(drop=True))
            st.write(moving_ave.shape)


    try:
        a = pd.read_csv('Filtered Dataset.csv')
        if a.equals(current_df):
            st.sidebar.warning("Currently saved results are equal to the current dataframe")
        else:
            st.dataframe(a.dtypes.astype(str))
            st.dataframe(current_df.dtypes.astype(str))
    except:
        st.sidebar.success("You haven't made any changes to the original dataframe yet")
    

    
    st.sidebar.subheader("Finalize smoothing & filtering changes:")
    if st.sidebar.button("Finalize!"):
        current_df.to_csv("Preprocessing dataset.csv", index=False)
        if os.path.isfile("Filtered Dataset.csv"):
            os.remove("Filtered Dataset.csv")
        st.sidebar.success("Saved!")
    else:
        st.sidebar.error("You have unsaved changes")

    # try:
    #     b = pd.read_csv('Preprocessing dataset.csv')
    #     if b.equals(current_df):
    #         st.sidebar.success("Your changes are finalized")
    #     else:
    #         st.sidebar.error("Your changes are not finalized.")
    # except:
    #     st.sidebar.warning("Bleh")



    # st.sidebar.subheader("Finalize changes:")
    # if st.sidebar.button("Finalize!"):
    #     current_df.to_csv("Preprocessing dataset.csv", index=False)
    #     if os.path.isfile("Filtered Dataset.csv"):
    #         os.remove("Filtered Dataset.csv")
    #     st.sidebar.success("Saved!")
    # else:
    #     st.sidebar.error("You have unsaved changes")

    # try:
    #     a = pd.read_csv('Filtered Dataset.csv')
    #     if a.equals(current_df):
    #         st.sidebar.write("Filtered equals current")
    #     else:
    #         st.dataframe(a.dtypes.astype(str))
    #         st.dataframe(current_df.dtypes.astype(str))
    # except:
    #     st.sidebar.write("")
