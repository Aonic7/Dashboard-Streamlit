# General import section
#from tkinter import E
import pandas as pd #to work with dataframes
import streamlit as st #streamlit backend
import os
import numpy as np

from scipy.signal import medfilt

# Visualization import section
from Visualization.visualization import doubleLinePlot, DoubleBoxPlot, Histogram, ScatterPlot, interpolation_subplot

# Remove outliers import
# from .smoothing_and_filtering_functions import removeOutlier, removeOutlier_q, removeOutlier_z
from .smoothing_and_filtering_functions import Remove_Outliers, Smoothing, TimeSeriesOOP, Converter

def import_dset(data_obj):
    try:
        a = pd.read_csv('Smoothing_and_Filtering//Filtered Dataset.csv', index_col = None)
        if a.equals(data_obj.df) == False:
            current_df1 = a
            current_df = Converter.dateTime_converter(current_df1)
            # st.sidebar.write("1")
            #st.sidebar.write(a.equals(data_obj.df))
        else:
            current_df = data_obj.df
            current_df = Converter.dateTime_converter(current_df)
            # st.sidebar.write("2")
    except:
        current_df = data_obj.df
        current_df = Converter.dateTime_converter(current_df)
        # st.sidebar.write("3")

    return current_df


def main(data_obj):
    st.header("Smoothing and filtering")

    # A button to circumvent loading the dataset using in the last session
    if st.sidebar.button("Reset dataframe to the initial one"):
        data_obj.df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)
        # st.dataframe(data_obj.df)
        #st.write(data_obj.df.shape)
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
                    rm_outlier = Remove_Outliers.removeOutlier(current_df, selected_column, std_coeff)
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
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

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
                    q_outlier = Remove_Outliers.removeOutlier_q(current_df, selected_column, q_values[0], q_values[1])
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
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

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
                    z_outlier = Remove_Outliers.removeOutlier_z(current_df, selected_column, std_coeff)
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
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)



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
                    median_filt = Smoothing.median_filter(current_df, selected_column, filter_len)
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

                if bp:
                    DoubleBoxPlot(data_obj.df, median_filt.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(median_filt.reset_index(drop=True), selected_column)

                if st.button("Save intermediate smoothing results (median filter)"):
                    current_df = median_filt.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        if smooth_radio == 'Moving average':
            with st.container():
                st.subheader('Moving average')

                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 1, 10, 5, 1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    # median_filt = median_filter(current_df, selected_column, filter_len)
                    moving_ave = Smoothing.moving_average(current_df, selected_column, filter_len)

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

                if bp:
                    DoubleBoxPlot(data_obj.df, moving_ave.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(moving_ave.reset_index(drop=True), selected_column)

                if st.button("Save intermediate smoothing results (moving average)"):
                    current_df = moving_ave.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)

        if smooth_radio == 'Savitzky Golay':
            with st.container():
                st.subheader('Savitzky Golay')

                cc1, cc2, cc3 = st.columns(3)

                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
                    filter_len = st.slider('Length of the window', 1, 11    , 5, 2)
                    order = st.slider('Polynomial order', 1, 3, 2, 1)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    sg = Smoothing.savitzky_golay(current_df, selected_column, filter_len, order)

                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    bp = st.button("Boxplot")
                    hist = st.button("Histogram")
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-sg.shape[0]} rows will be removed.')
                
                if plot_basic:
                    doubleLinePlot(data_obj.df, sg.reset_index(drop=True), selected_column)

                if bp:
                    DoubleBoxPlot(data_obj.df, sg.reset_index(drop=True), selected_column)
                if hist:
                    Histogram(sg.reset_index(drop=True), selected_column)

                if st.button("Save intermediate smoothing results (Savitzky Golay)"):
                    current_df = sg.reset_index(drop=True)
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)


    if dp_method == 'Interpolation':
        interpolation_radio = st.radio(label = 'Interpolation',
                             options = ['Linear','Cubic', 'Forward Fill', 'Backward Fill'])
        # if interpolation_radio == 'All':
            
        #     with st.container():
        #         st.subheader('All interpolations')

        #         cc1, cc2, cc3 = st.columns(3)
        #         with cc1:
        #             columns_list = list(current_df.select_dtypes(exclude=['object']).columns)
        #             columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
        #             selected_column = st.selectbox("Select a column:", columns_list)
        #             time_column = st.selectbox("Select a time column:", columns_list1)
        #             interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column)
                    
        #         with cc2:
        #             st.write(" ")
        #             st.write(" ")
        #             plot_basic = st.button('Plot')
        #             #bp = st.button("Boxplot")
        #             #hist = st.button("Histogram")
        #         # with cc3:
        #         #     st.write(" ")
        #         #     st.write(" ")
        #         #     st.warning(f'If applied, {current_df.shape[0]-median_filt.shape[0]} rows will be removed.')
                
        #         if plot_basic:
        #             interpolation_all.draw_all(selected_column) 
        #             #doubleLinePlot(data_obj.df, interpolation_all.draw_all(selected_column), selected_column)
        #             # st.dataframe(median_filt)
        #             # st.write(data_obj.df[selected_column].value_counts(ascending=False))
        #             # st.write(median_filt[selected_column].value_counts(ascending=False))
        #             #st.write("Blah")
        #             # l = {'col1': medfilt(data_obj.df[selected_column], filter_len)}
        #             # lf = pd.DataFrame(data=l)
        #             # st.write(lf.value_counts(ascending=False))


        #         #if bp:
        #             #DoubleBoxPlot(data_obj.df, interpolation_all.draw_all().reset_index(drop=True), selected_column)

        #         #if hist:
        #             #Histogram(interpolation_all.draw_all().reset_index(drop=True), selected_column)

        #         #current_df = rm_outlier.reset_index(drop=True)
    
        if interpolation_radio == 'Linear':
                 
            with st.container():
                st.subheader('Linear interpolation')

                cc1, cc2, cc3 = st.columns(3)
            
                try:
                    with cc1:
                        columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                        columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                        selected_column = st.selectbox("Select a column:", columns_list)
                        time_column = st.selectbox("Select a time column:", columns_list1)
                        interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column)
                        linear_df = interpolation_all.make_interpolation_liner(selected_column) 
                    with cc2:
                        st.write(" ")
                        st.write(" ")
                        plot_basic = st.button('Plot')
                        
                    with cc3:
                        st.write(" ")
                        st.write(" ")
                        st.warning(f'If applied, {current_df.shape[0]-linear_df.shape[0]} rows will be removed.')
                        
                    if plot_basic:
                            interpolation_subplot(data_obj.df, linear_df, selected_column, 'linear_fill')
                        
                    if st.button("Save intermediate linear results"):
                            current_df = linear_df.reset_index()
                            current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)  
                except KeyError as e:
                        st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                        st.stop()   

                    
    
        if interpolation_radio == 'Cubic':
            
            with st.container():
                st.subheader('Cubic interpolation')
            try:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column)
                    Cubic_df = interpolation_all.make_interpolation_cubic(selected_column) 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                
                with cc3:
                     st.write(" ")
                     st.write(" ")
                     st.warning(f'If applied, {current_df.shape[0]-Cubic_df.shape[0]} rows will be removed.')
                
                if plot_basic: 
                   interpolation_subplot(current_df, Cubic_df, selected_column, 'cubic_fill')
    
                if st.button("Save intermediate cubic results"):
                    current_df = Cubic_df.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)   
            except KeyError as e: 
                   st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                   st.stop()        
        
        if interpolation_radio == 'Forward Fill':
            
            with st.container():
                st.subheader('Forward Fill interpolation')
            try:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column)
                    df_ffill = interpolation_all.int_df_ffill() 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
              
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-df_ffill.shape[0]} rows will be removed.')
                
                if plot_basic:
                   interpolation_subplot(current_df, df_ffill, selected_column, 'Forward Fill')
                
                if st.button("Save intermediate Forward Fill results"):
                    current_df = df_ffill.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False) 
            except KeyError as e: 
                 st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                 st.stop()   

        if interpolation_radio == 'Backward Fill':
            
            with st.container():
                st.subheader('Backward Fill interpolation')
            try:
                cc1, cc2, cc3 = st.columns(3)
                with cc1:
                    columns_list = list(current_df.select_dtypes(exclude=['datetime', 'object']).columns)
                    columns_list1 = list(current_df.select_dtypes(include=['datetime']).columns)
                    selected_column = st.selectbox("Select a column:", columns_list)
                    time_column = st.selectbox("Select a time column:", columns_list1)
                    interpolation_all = TimeSeriesOOP(current_df, selected_column, time_column)
                    df_bfill = interpolation_all.int_df_bfill() 
                    
                with cc2:
                    st.write(" ")
                    st.write(" ")
                    plot_basic = st.button('Plot')
                    
                with cc3:
                    st.write(" ")
                    st.write(" ")
                    st.warning(f'If applied, {current_df.shape[0]-df_bfill.shape[0]} rows will be removed.')
                
                if plot_basic: 
                   interpolation_subplot(current_df, df_bfill, selected_column, 'Backward Fill')
                
                if st.button("Save intermediate Backward Fill results"):
                    current_df = df_bfill.reset_index()
                    current_df.to_csv("Smoothing_and_Filtering//Filtered Dataset.csv", index=False)           
            except KeyError as e:
                st.warning("Selected dataframe is not appropriate for this method, please upload a different one")
                st.stop()   
                
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
        if dp_method == 'Smoothing' and smooth_radio == 'Savitzky Golay':
            st.dataframe(sg.reset_index(drop=True))
            st.write(sg.shape)

        # For each 'interpolation' case   
        if dp_method == 'Interpolation' and  interpolation_radio == 'Linear':
            st.dataframe(linear_df.reset_index())
            st.write(linear_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Cubic':
            st.dataframe(Cubic_df.reset_index())
            st.write(Cubic_df.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Forward Fill':
            st.dataframe(df_ffill.reset_index())
            st.write(df_ffill.shape)
        if dp_method == 'Interpolation' and  interpolation_radio == 'Backward Fill':
            st.dataframe(df_bfill.reset_index())
            st.write(df_bfill.shape)          


    try:
        a = pd.read_csv("Smoothing_and_Filtering//Filtered Dataset.csv")
        if a.equals(current_df):
            st.sidebar.warning("Currently saved results are equal to the current dataframe")
    except:
        st.sidebar.success("You haven't made any changes to the original dataframe yet")
    

    
    st.sidebar.subheader("Finalize smoothing & filtering changes:")
    if st.sidebar.button("Finalize!"):
        current_df.to_csv("Smoothing_and_Filtering//Preprocessing dataset.csv", index=False)
        if os.path.isfile("Smoothing_and_Filtering//Filtered Dataset.csv"):
            os.remove("Smoothing_and_Filtering//Filtered Dataset.csv")
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

if __name__ == "__main__":
    main()
